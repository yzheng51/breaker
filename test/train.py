# =========================================================================================
# Libraries
# =========================================================================================
import os
import time
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils.loss import kl_divergence_loss
from utils.evaluate import uauc_score
from utils.dataset import MyDataset
from utils.model import UserRepresentation, Breaker

# =========================================================================================
# Seed everything for deterministic results
# =========================================================================================
def seed_everything(seed_value):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use {device}")

# =========================================================================================
# Functions
# =========================================================================================
def get_cluster_center(model, train_loader, n_clusters=4):
    # get user embed
    model.eval()
    user_embed_ls = list()
    for u_feat, _, _ in train_loader:
        u_feat = u_feat.long().to(device)
        with torch.no_grad():
            user_embed = model(u_feat)

        user_embed_ls.append(user_embed.cpu().numpy())

    user_embed_ls = np.vstack(user_embed_ls)

    del u_feat, user_embed

    # get cluster center
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans = kmeans.fit(user_embed_ls)

    return kmeans.cluster_centers_

# =========================================================================================
# Data Loading
# =========================================================================================
input_train_file_path = "../data/train.parquet"
input_valid_file_path = "../data/valid.parquet"
feat_info_file_path = "../data/feat_info.parquet"
label_col = "label"

train = pd.read_parquet(input_train_file_path)
valid = pd.read_parquet(input_valid_file_path)
'''
# data frame looks like below, the value of u_feat_x range from 0 to 10 (the max value can be less than 10)
| dt       | uid | label | item   | u_feat_1 | u_feat_2 | u_feat_3 | u_feat_4 | u_feat_5 |
| -------- | --- | ----- | ------ | -------- | -------- | -------- | -------- | -------- |
| 19990101 | u1  | 1     | item_1 | 1        | 2        | 3        | 4        | 5        |
| 19990102 | u2  | 0     | item_2 | 2        | 3        | 4        | 5        | 6        |
'''

item_mapping = {
    "item_1": 0,
    "item_2": 1,
    "item_3": 2,
    "item_4": 3,
    "item_5": 4,
    "item_6": 5,
    "item_7": 6,
    "item_8": 7,
    "item_9": 8,
    "item_10": 9
}
candidates = list(item_mapping.keys())

feats = [
    "u_feat_1",
    "u_feat_2",
    "u_feat_3",
    "u_feat_4",
    "u_feat_5",
]
print(f"len(feats)={len(feats)}")
print("train.columns")
print(train.columns[:5])
print(train.columns[-5:])

# =========================================================================================
# Create dataset for computing GAUC
# =========================================================================================
uauc_valid = valid.loc[valid[label_col] > 0].reset_index(drop=True).copy()
print(f"uauc_valid.shape={uauc_valid.shape}")

to_valid = uauc_valid[["uid", "dt"] + feats].drop_duplicates().reset_index(drop=True)
to_valid["item"] = ",".join(map(str, candidates))
to_valid["item"] = to_valid["item"].str.split(",")

to_valid = to_valid.explode("item")
to_valid["item"] = to_valid["item"].astype("int32")

tmp = uauc_valid[["uid", "item", "dt", label_col]]

to_valid = to_valid.merge(tmp, on=["uid", "item", "dt"], how="left")
to_valid[label_col].fillna(0, inplace=True)
to_valid[label_col] = to_valid[label_col].astype("int8")

uauc_valid = to_valid.copy()
del to_valid, tmp

print(f"uauc_valid.shape={uauc_valid.shape}")

# =========================================================================================
# Negative sampling
# =========================================================================================
pos = train.loc[train[label_col] == 1]
neg = train.loc[train[label_col] == 0].sample(frac=0.2)
train = pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)

print("After negative sampling")
print(train.info())

# =========================================================================================
# Prepare for NN
# =========================================================================================
feat_info = pd.read_parquet(feat_info_file_path)
feat_map = dict(zip(feat_info["feat"], feat_info["counter"]))
'''
feat_map is a mapping from a user feat to its unique count which looks like
{
  "u_feat_1": 10,
  "u_feat_2": 10,
  "u_feat_3": 9,
  "u_feat_4": 8,
  "u_feat_5": 5,
}
'''

train["item"] = train["item"].map(item_mapping)
valid["item"] = valid["item"].map(item_mapping)
uauc_valid["item"] = uauc_valid["item"].map(item_mapping)

# cumsum for each feat to ensure it can be embedded in one class
counter = 0
for f in feats:
    train[f] = train[f].values + counter
    valid[f] = valid[f].values + counter
    uauc_valid[f] = uauc_valid[f].values + counter

    counter += feat_map[f]

print(f"counter={counter}")

train_user_feat = train[feats].values
valid_user_feat = valid[feats].values
uauc_valid_user_feat = uauc_valid[feats].values

print(
    f"train_user_feat.shape={train_user_feat.shape}, "
    f"valid_user_feat.shape={valid_user_feat.shape}, "
    f"uauc_valid_user_feat.shape={uauc_valid_user_feat.shape}"
)

y_train = train[[label_col]].values
y_valid = valid[[label_col]].values

train_data = MyDataset(train_user_feat, train[["item"]].values, y_train)
valid_data = MyDataset(valid_user_feat, valid[["item"]].values, y_valid)
uauc_valid_data = MyDataset(uauc_valid_user_feat, uauc_valid[["item"]].values, uauc_valid[[label_col]].values)

train_loader = torch.utils.data.DataLoader(
    train_data,
    shuffle=True,
    drop_last=True,
    batch_size=2048 * 2,
    num_workers=2
)
valid_loader = torch.utils.data.DataLoader(
    valid_data,
    shuffle=False,
    drop_last=False,
    batch_size=2048 * 4,
    num_workers=2
)
uauc_valid_loader = torch.utils.data.DataLoader(
    uauc_valid_data,
    shuffle=False,
    drop_last=False,
    batch_size=2048 * 4,
    num_workers=2
)

# =========================================================================================
# Train Breaker
# =========================================================================================
epochs = 20
n_clusters = 4
embed_size = 5
# user representation
init_user_embed = UserRepresentation(
    n_feat=counter,
    n_field=len(feats),
    embed_size=embed_size
)
# find cluster center
init_user_embed.to(device)
init_cluster_centers = get_cluster_center(init_user_embed, train_loader, n_clusters=n_clusters)
# network
net = Breaker(
    n_feat=counter,
    n_field=len(feats),
    n_clusters=n_clusters,
    init_user_embed=init_user_embed,
    init_cluster_centers=init_cluster_centers,
    embed_size=embed_size
)
net.to(device)
# network for p calculation
net_for_p = Breaker(
    n_feat=counter,
    n_field=len(feats),
    n_clusters=n_clusters,
    init_user_embed=init_user_embed,
    init_cluster_centers=init_cluster_centers,
    embed_size=embed_size
)
net_for_p.to(device)
net_for_p.load_state_dict(net.state_dict())

criterion_kl = kl_divergence_loss
criterion_log = torch.nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

print("|  Epoch  |     Step     |   Score    |    GAUC    |  Recall@1  | Epapsed Time |")
print("| ------- | ------------ | ---------- | ---------- | ---------- | ------------ |")
for e in range(epochs):
    start = time.perf_counter()
    train_kl_losses = list()
    train_log_losses = list()
    train_losses = list()
    net.train()
    for ii, (u_feat, i_feat, labels) in enumerate(train_loader):
        u_feat, i_feat, labels = u_feat.long().to(device), i_feat.long().to(device), labels.float().to(device)

        output_q, output_target = net(u_feat, i_feat)
        # get p value without gradient
        net_for_p.eval()
        with torch.no_grad():
            output_p, _ = net_for_p(u_feat, i_feat)
            output_p = output_p.square() / output_p.sum(0)
            output_p = output_p / output_p.sum(axis=1, keepdims=True)

        kl_loss = criterion_kl(output_q, output_p)
        log_loss = criterion_log(output_target.view(-1, 1), labels)
        loss = 0.1 * kl_loss + log_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())
        train_kl_losses.append(kl_loss.item())
        train_log_losses.append(log_loss.item())

        if (ii + 1) % 100 == 0:
            net_for_p.load_state_dict(net.state_dict())
            print(np.mean(train_kl_losses))
            print(np.mean(train_log_losses))
            print(np.mean(train_losses))

    net.eval()
    test_score = list()
    for u_feat, i_feat, labels in valid_loader:
        u_feat, i_feat, labels = u_feat.long().to(device), i_feat.long().to(device), labels.float().to(device)
        with torch.no_grad():
            output_q, output_target = net(u_feat, i_feat)

        test_score.append(output_target.cpu().numpy())
    test_score = np.hstack(test_score)

    net.eval()
    test_score_uauc = list()
    for u_feat, i_feat, labels in uauc_valid_loader:
        u_feat, i_feat, labels = u_feat.long().to(device), i_feat.long().to(device), labels.float().to(device)
        with torch.no_grad():
            output_q, output_target = net(u_feat, i_feat)

        test_score_uauc.append(output_target.cpu().numpy())
    test_score_uauc = np.hstack(test_score_uauc)

    # output train information
    uauc = uauc_score(uauc_valid[label_col], test_score_uauc.ravel(), uauc_valid["uid"].values)
    uauc_valid["score"] = test_score_uauc.ravel()
    recall_at_1 = uauc_valid.sort_values(["uid", "score", "item"], ascending=False).reset_index(drop=True)
    recall_at_1["item"] = recall_at_1["item"].map(dict(zip(list(item_mapping.values()), list(item_mapping.keys()))))
    print(recall_at_1.groupby("uid").head(1)["item"].value_counts())
    recall_at_1 = recall_at_1.groupby("uid").head(1)[label_col].mean()
    print(output_q.mean(0))
    print(pd.Series(output_q.argmax(1).cpu().numpy()).value_counts())
    duration = time.perf_counter() - start
    print(
        "|  {}/{} ".format(e + 1, epochs).ljust(9),
        "|  {}/{} ".format(ii, len(train_loader)).ljust(14),
        "|   {:.4f}  ".format(roc_auc_score(y_valid, test_score)).ljust(12),
        "|   {:.4f}  ".format(uauc).ljust(12),
        "|   {:.4f}  ".format(recall_at_1).ljust(12),
        "|  {:.2f}s ".format(duration).ljust(14),
        "|"
    )
    model_path = f"../models/breaker/{e+1}.pth"
    torch.save(net.state_dict(), model_path)

# =========================================================================================
# Visualization
# =========================================================================================
model_path = "../models/breaker/10.pth"
net = Breaker(
    n_feat=counter,
    n_field=len(feats),
    n_clusters=n_clusters,
    init_user_embed=init_user_embed,
    init_cluster_centers=init_cluster_centers,
    embed_size=embed_size
)
net.to(device)
net.load_state_dict(torch.load(model_path))

net.eval()
test_score = list()
dev_rep = list()
dev_clus = list()
for u_feat, i_feat, labels in valid_loader:
    u_feat, i_feat, labels = u_feat.long().to(device), i_feat.long().to(device), labels.float().to(device)
    with torch.no_grad():
        output_q, output_target = net(u_feat, i_feat)
        output_user_rep = net.user_embed(u_feat)

        dev_rep.append(output_user_rep.cpu().numpy())
        dev_clus.append(output_q.cpu().numpy())

    break

dev_rep = np.vstack(dev_rep)
dev_clus = np.vstack(dev_clus)

dev_clus = dev_clus.argmax(1)
print(pd.Series(dev_clus).value_counts())

tsne = TSNE(n_components=2,init='random')
X_embed = tsne.fit_transform(dev_rep)

plt.figure()
plt.scatter(X_embed[:,0], X_embed[:, 1], s=1, c=dev_clus)
