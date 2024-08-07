import json
import logging
import os
import time
from collections import Counter
from shutil import copyfile

import configargparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras.utils import GeneratorEnqueuer
from tqdm import tqdm


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    if os.path.exists(log_path):
        os.remove(log_path)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

class Breaker(object):
    def __init__(self, embedding_dim, learning_rate,
                target_cycle=1000, verbose=1, lamda=1e-6, alpha=1,earlystop_patience=1,
                 encode_dims=[256, 128, 64], tower_layers=[32, 10], n_cluster=6, cluster_alpha=1,random_seed=2023):

        ## Cluster
        self.alpha = alpha
        self.n_cluster = n_cluster
        self.cluster_alpha = cluster_alpha
        self.kmeans = KMeans(n_clusters=self.n_cluster)

        ## training
        self.target_cycle = target_cycle
        self.learning_rate = learning_rate

        ## early stop
        self.earlystop_patience = earlystop_patience
        self.best_aer = -999
        self.earlystop_counter = 0

        ## network structure
        self.encode_dims = encode_dims
        self.tower_layers = tower_layers
        self.embedding_dim = embedding_dim
        self.lamda = lamda

        self.random_seed = random_seed
        self.verbose = verbose

        # define model
        self.build_placeholder()

        self.build_dense_weight()

        self.build_input_layer()

        self.define_loss()

    def build_dense_weight(self):
        all_weights = dict()
        l2_reg = tf.contrib.layers.l2_regularizer(self.lamda)

        with tf.name_scope("user_embedding"):
            for column in feat_set:
                all_weights['feature_embeddings_{}'.format(column)] = tf.get_variable(
                    initializer=tf.random_normal(
                        shape=[
                            feat_box_info[column],
                            self.embedding_dim],
                        mean=0.0,
                        stddev=0.01),
                    regularizer=l2_reg, name='feature_embeddings_{}'.format(column))

        with tf.name_scope("item_embedding"):
            self.treatment_weight = tf.get_variable(
                initializer=tf.random_normal(
                    shape=[
                        feat_box_info['if_0'],
                        self.embedding_dim],
                    mean=0.0,
                    stddev=0.01),
                regularizer=l2_reg, name='feature_embeddings_{}'.format('treatment'))

        self.weights = all_weights


    def build_input_layer(self):
        feature_embedding = []

        for column, feature in zip(feat_set, self.feat_inputs_placeholder):
            embedded = tf.squeeze(tf.nn.embedding_lookup(
                self.weights['feature_embeddings_{}'.format(column)], feature), axis=1)
            feature_embedding.append(embedded)

        self.input_layer = tf.keras.layers.concatenate(
            feature_embedding, axis=-1)

        self.item_embedding = tf.squeeze(tf.nn.embedding_lookup(
            self.treatment_weight, self.item_placeholder), axis=1)


    # build placeholder
    def build_placeholder(self):
        self.feat_inputs_placeholder = []

        self.train_label = tf.placeholder(
            tf.float32, shape=[None, 1], name='label')

        # define feature placeholder
        for column in feat_set:
            self.feat_inputs_placeholder.append(tf.placeholder(
                tf.int32, shape=[None, 1], name=column))

        self.item_placeholder = tf.placeholder(
            tf.int32, shape=[None, 1], name='material_id')

    def get_wholetensor(self, dev_df, typ='usrrep'):
        idx_start = np.linspace(0, len(dev_df), dtype=int, num=50)[:-1]
        idx_end = np.linspace(0, len(dev_df), dtype=int, num=50)[1:]

        encoder_ls = []
        for start, end in zip(idx_start, idx_end):
            feed_dict = {}
            for column_name, column_placeholder in zip(feat_set, self.feat_inputs_placeholder):
                feed_dict[column_placeholder] = np.expand_dims(
                    dev_df[column_name][start:end].values, axis=1)
            if typ == 'usrrep':
                encoder = self.sess.run(self.user_rep, feed_dict=feed_dict)
            elif typ == 'q':
                encoder = self.sess.run(self.q, feed_dict=feed_dict)
            elif typ == 'cluspred':
                encoder = self.sess.run(self.cluster_pred, feed_dict=feed_dict)
            else:
                raise NameError('No {} type'.format(typ))
            encoder_ls.append(encoder)
        encoder = np.concatenate(encoder_ls, axis=0)
        return encoder

    def build_dense(self, inp_shape, layers, end_activation=False):
        """
        params:
            - units_num: user layers structure
        """
        inp = tf.keras.layers.Input(
            shape=(inp_shape,), name='input_layer')

        if not end_activation:
            for layer_idx, layer_num in enumerate(layers):
                if layer_idx == 0:
                    oup = tf.keras.layers.Dense(layer_num, activation='relu')(inp)
                elif layer_idx != len(layers) - 1:
                    oup = tf.keras.layers.Dense(layer_num, activation='relu')(oup)
                else:
                    oup = tf.keras.layers.Dense(layer_num)(oup)
                oup = tf.keras.layers.Dropout(0.5)(oup)
        else:
            for layer_idx, layer_num in enumerate(layers):
                if layer_idx == 0:
                    oup = tf.keras.layers.Dense(layer_num, activation='relu')(inp)
                else:
                    oup = tf.keras.layers.Dense(layer_num, activation='relu')(oup)
                oup = tf.keras.layers.Dropout(0.5)(oup)

        model = tf.keras.Model(inputs=inp, outputs=oup)
        return model

    def get_assign_cluster_centers_op(self, features):
        # init mu
        logger.info("Kmeans train start.")
        kmeans = self.kmeans.fit(features)
        logger.info("Kmeans train end.")
        return tf.assign(self.mu, kmeans.cluster_centers_)

    def _soft_assignment(self, embeddings, cluster_centers):
        """Implemented a soft assignment as the  probability of assigning sample i to cluster j.

        Args:
            embeddings: (num_points, dim)
            cluster_centers: (num_cluster, dim)

        Return:
            q_i_j: (num_points, num_cluster)
        """

        def _pairwise_euclidean_distance(a, b):
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(tf.shape(a)[0], 1)),
                transpose_b=True
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 *
                          tf.matmul(a, b, transpose_b=True))
            return res

        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0 / (1.0 + dist ** 2 / self.alpha) ** ((self.alpha + 1.0) / 2.0)
        q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def _kl_divergence(self, target, pred):
        return tf.reduce_mean(tf.reduce_sum(target * tf.log(target / (pred)), axis=1))

    def build_train_gen(self, df, max_q_size=5, workers=1, batch_size=2048):
        data_gen = self.iterator_pandas(df, batch_size=batch_size)
        enqueuer = GeneratorEnqueuer(data_gen, use_multiprocessing=True)
        enqueuer.start(max_queue_size=max_q_size, workers=workers)
        return enqueuer

    def iterator_pandas(self, train_df, batch_size):
        length = len(train_df)
        idx = np.arange(0, length, batch_size)
        batch_data = {}
        for start, end in zip(idx[:-1], idx[1:]):
            df = train_df.iloc[start:end, :]
            batch_data['label'] = df["label"].values  # appliy
            for col in feat_set:
                batch_data[col] = np.expand_dims(df[col].values, axis=1)

            batch_data['item'] = np.expand_dims(
                df['if_0'].values, axis=1)
            batch_data['idx'] = np.arange(start, end)
            yield batch_data
        yield None

    def train(self, epochs, batch_size):
        global train_df

        # initialize cluster centers (based on k-means)
        z = self.get_wholetensor(train_df, typ='usrrep')
        print('z shape:', z.shape)

        assign_mu_op = self.get_assign_cluster_centers_op(z)
        _ = self.sess.run(assign_mu_op)

        start = time.perf_counter()
        train_loss, train_logloss, train_clusloss = 0, 0, 0

        iter_global = 0
        for epoch in range(epochs):
            if epoch != 0:
                # shuffle data
                train_df = train_df.sample(frac=1).reset_index(drop=True)

            dec_ckpt_epoch_path = os.path.join(
                model_save_path, 'dec_ckpt_epoch_{}'.format(epoch), 'model.ckpt')

            tf.keras.backend.set_learning_phase(1)
            logger.info('train learning_phase:{}'.format(tf.keras.backend.learning_phase()))

            enqueuer = self.build_train_gen(train_df, batch_size=batch_size)
            for iter_, generator_output in enumerate(enqueuer.get()):
                iter_global += 1
                if generator_output is None:
                    break
                else:
                    if iter_ % self.target_cycle == 0:
                        q = self.get_wholetensor(train_df, typ='q')
                        self.target_p = self.target_distribution(q)

                    loss_temp, q_mean, p_mean, q, cluster_pred, usrrep = self.train_on_batch(generator_output)
                    train_loss, train_logloss, train_clusloss = train_loss + loss_temp[0], train_logloss + loss_temp[
                        1], train_clusloss + loss_temp[2]
                    if iter_ % 50 == 0:
                        logger.info(
                            '[%d]Train loss on step %d: %.6f, logloss: %.6f, clustloss: %.6f [use time per iter:%.3f]' % (
                                iter_ * batch_size, iter_, \
                                train_loss / (iter_global), \
                                train_logloss / (iter_global), \
                                train_clusloss / (iter_global), \
                                (time.perf_counter() - start) / (iter_global)))

                        logger.info(q_mean)
                        logger.info(p_mean)
                        logger.info(Counter(cluster_pred))

            start2 = time.perf_counter()
            # early stop
            tf.keras.backend.set_learning_phase(0)
            aer_score = self.evaluate_aer(dev_df)
            if aer_score > self.best_aer:
                print(f'Validation AER increased ({self.best_aer:.6f} --> {aer_score:.6f}).  Saving model ...')
                self.best_aer = aer_score
                self.saver.save(self.sess, dec_ckpt_epoch_path)
            else:
                self.earlystop_counter+=1
                print(f'Validation AER decreased ({self.best_aer:.6f} --> {aer_score:.6f}).')
                if self.earlystop_counter >= self.earlystop_patience:
                    print(f'EarlyStopping counter: {self.earlystop_counter} out of {self.earlystop_patience};')
                    break

            print('valid time:%.3f [use time per epoch:%.3f]' % (
                time.perf_counter() - start2, (time.perf_counter() - start2) / (epoch + 1)))

            enqueuer.stop()

    def define_loss(self):
        self.graph = tf.Graph()
        tf.set_random_seed(self.random_seed)

        # user representation
        with tf.name_scope("user_rep"):
            feature_embedding = tf.concat([self.input_layer], axis=-1)
            featre_extractor = self.build_dense(len(feat_set) * self.embedding_dim, self.encode_dims,
                                                end_activation=True)
            self.user_rep = featre_extractor(feature_embedding)


        with tf.name_scope("ui_tower"):
            tower_ls = [self.build_dense(self.encode_dims[-1] + self.embedding_dim, self.tower_layers) for i in
                        range(self.n_cluster)]
            tower_input = tf.concat([self.user_rep, self.item_embedding], axis=-1)
            pred_ls = []
            for idx, tower_nn in enumerate(tower_ls):
                tower_out = tower_nn(tower_input)
                tower_out = tf.sigmoid(tower_out)
                pred_ls.append(tower_out)
            final_out = tf.concat(pred_ls, axis=-1)
            self.final_out = final_out

        # clustering
        self.mu = tf.Variable(
            tf.zeros(shape=(self.n_cluster, self.encode_dims[-1])), name="mu")
        with tf.name_scope("distribution"):
            self.q = self._soft_assignment(self.user_rep, self.mu)
            self.p = tf.placeholder(tf.float32, shape=(None, self.n_cluster))
            self.cluster_pred = tf.argmax(self.q, axis=1)

        self.apply = tf.reduce_sum(
            tf.multiply(final_out, self.q), axis=-1, keepdims=True)
        self.q_mean = tf.reduce_mean(self.q, axis=0)
        self.p_mean = tf.reduce_mean(self.p, axis=0)

        with tf.name_scope("cluster_loss"):
            self.cluster_loss = self._kl_divergence(self.p, self.q)

        with tf.name_scope("log_loss"):
            self.logloss = tf.losses.log_loss(self.train_label, self.apply)

        with tf.name_scope("union_loss"):
            self.loss = self.cluster_alpha * self.cluster_loss + self.logloss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)

        # init
        self.saver = tf.train.Saver(var_list=tf.global_variables())
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.InteractiveSession(config=config)

        self.sess.run(init)

        # number of params
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            logger.info("#params: %d" % total_parameters)

    def train_on_batch(self, data):
        train_ids = {}
        for column_name, column_placeholder in zip(feat_set, self.feat_inputs_placeholder):
            train_ids[column_placeholder] = data[column_name]

        train_ids[self.item_placeholder] = data['item']

        feed_dict = {self.train_label: np.expand_dims(data['label'], axis=1)}
        feed_dict.update(train_ids)

        feed_dict.update({self.p: self.target_p[data['idx']]})

        loss, q_mean, p_mean, q, cluster_pred, usrrep, _ = self.sess.run(
            ([self.loss, self.logloss, self.cluster_loss], self.q_mean, self.p_mean, self.q, self.cluster_pred,
             self.user_rep, self.optimizer), feed_dict=feed_dict)

        return loss, q_mean, p_mean, q, cluster_pred, usrrep

    def evaluate_debug(self):
        idx_start = np.linspace(0, len(dev_df), dtype=int, num=50)[:-1]
        idx_end = np.linspace(0, len(dev_df), dtype=int, num=50)[1:]

        pred_temp = []
        for start, end in zip(idx_start, idx_end):
            feed_dict = {}
            for column_name, column_placeholder in zip(feat_set, self.feat_inputs_placeholder):
                feed_dict[column_placeholder] = np.expand_dims(dev_df[column_name][start:end].values, axis=1)

            feed_dict[self.item_placeholder] = np.expand_dims(dev_df['if_0'][start:end].values, axis=1)
            predictions = self.sess.run(self.apply, feed_dict=feed_dict)
            pred_temp.append(predictions)

        pred = np.concatenate(pred_temp, axis=0)
        return {'apply_true': dev_df['label'],
                'apply_pred': pred}

    def get_full(self, dev_df, chunck_num):
        idx_start = np.linspace(0, len(dev_df), dtype=int, num=chunck_num)[:-1]
        idx_end = np.linspace(0, len(dev_df), dtype=int, num=chunck_num)[1:]

        prob_full = {}

        for item, item_code in tqdm(zip(candidate_item, candidate_item_code)):
            df_incre = []
            for start, end in zip(idx_start, idx_end):
                feed_dict = {}
                for column_name, column_placeholder in zip(feat_set, self.feat_inputs_placeholder):
                    feed_dict[column_placeholder] = np.expand_dims(dev_df[column_name][start:end].values, axis=1)

                feed_dict[self.item_placeholder] = np.ones([len(dev_df[column_name][start:end].values), 1]) * item_code

                incre_temp = self.sess.run(self.apply, feed_dict=feed_dict)
                df_incre.append(incre_temp)

            df_incre = np.concatenate(df_incre, axis=0)
            prob_full[item] = df_incre.ravel()
        return pd.DataFrame(prob_full)

    def evaluate_aer(self, df, chunck_num=20):

        # calculate AER
        prob_full = self.get_full(df, chunck_num)

        dec_idx = np.array(prob_full).argmax(axis=1)
        dec_idx = [candidate_item_code[i] for i in dec_idx]
        match_idx = dec_idx == df.if_0
        print(Counter(df.loc[match_idx, 'if_0']))

        aer = np.mean(df.loc[match_idx, 'label'])

        return aer

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add("--gpu-index", dest="gpu_index", help="GPU Index Number", default="0", type=str)
    parser.add('--suffix', dest="suffix", type=str, default='default')
    parser.add('--file', dest="file", type=str, default='default')


    root_dir = './' #TODO: replace your absolute path
    train_path = os.path.join(root_dir, 'data',
                              'train_data_fornn.csv')
    valid_path = os.path.join(root_dir, 'data',
                              'test_data_fornn.csv')

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(valid_path)

    args = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_index']
    file_path = args['file']
    suffix = args['suffix']

    curtime = time.strftime('%m%d%H%M', time.localtime(time.time()))
    model_log_path = os.path.join(root_dir, 'log', str(curtime) + '_' + suffix)
    mkdir(model_log_path)
    print('model_log_path:{}'.format(model_log_path))
    logger = logger_config(model_log_path + '/train.log', '-')

    model_save_path = model_log_path + '/' + 'model_file'
    mkdir(model_save_path)
    if file_path != 'default':
        copyfile(file_path, os.path.join(model_save_path, 'script.txt'))

    # user-related features used by the model
    feat_set = ['uf_0', 'uf_1', 'uf_2', 'uf_3', 'uf_4', 'uf_5', 'uf_6', 'uf_7',
       'uf_8', 'uf_9', 'uf_10', 'uf_11', 'uf_12', 'uf_13', 'uf_14', 'uf_15',
       'uf_16']
    feat_box_info = {}
    with open(os.path.join(root_dir, 'config', 'feat_info.json'), 'r') as f:
        info = json.load(f)
    coltype = info['colType']
    encode_info = info['encode']
    bin_info = info['manualBox']

    feat_box_info['if_0'] = len(
        encode_info['if_0']) + 2
    for col in feat_set:
        if coltype[col] == "E":
            feat_box_info[col] = len(encode_info[col]) + 2
        else:
            feat_box_info[col] = len(bin_info[col]) + 2

    # params
    params = {"tower_layers": [32, 10, 1],
              "embedding_dim": 10,
              "n_cluster": 4,
              "learning_rate": 1e-3,
              "cluster_alpha": 0.1,
              "encode_dims": [256, 64],
              "epochs": 10,
              "batch_size": 128,
              "alpha": 1,
              "earlystop_patience": 1}

    steps_per_epoch = int(len(train_df)/params['batch_size'])
    target_cycle = int(steps_per_epoch * 0.1)
    params['target_cycle'] = target_cycle

    candidate_item = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    candidate_item_code = [encode_info['if_0'][str(int(i))] for i in candidate_item]

    tf.reset_default_graph()
    model = Breaker(embedding_dim=params['embedding_dim'], tower_layers=params['tower_layers'],
                encode_dims=params['encode_dims'], n_cluster=params['n_cluster'], alpha=params['alpha'],
                learning_rate=params['learning_rate'], cluster_alpha=params['cluster_alpha'],
                target_cycle=params['target_cycle'],earlystop_patience=params['earlystop_patience'])
    logger.info('param:{}'.format(params))
    model.train(epochs=params['epochs'], batch_size=params['batch_size'])