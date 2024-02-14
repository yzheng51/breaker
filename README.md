# Breaker: Removing Shortcut Cues with User Clustering for Single-slot Recommendation System

This repo contains the implementation of Breaker in Tensorflow and PyTorch.

## Directory Structure Explanation

```shell
├── README.md
├── config # configuration file for data preprocessing
│   └── feat_info.json
├── data # data file
│   ├── demo_data.csv
│   ├── test_data_fornn.csv
│   └── train_data_fornn.csv
├── log # run log
├── requirements.txt # environment
└── script
    ├── breaker_tf.py  # implemented with tensorflow
    ├── breaker_torch # implemented with Torch
    └── data_preprocessing.ipynb # data preprocessing
```

## How to run? 

### 1 Prepare the Environment

```python
pip install -r requirements.txt
```

### 2 Data Preprocessing

You will need to prepare a configuration file for data preprocessing as shown in feat_info.json, which includes three fields:

- colType: Specifies whether a feature is categorical (denoted by 'E') or continuous (denoted by 'M').
- encode: For categorical features, specify the encoding method for embedding purposes.
- manualBox: For continuous features, specify the feature value split points to convert them into discrete features.

An example dataset, demo_data.csv, is stored in the data folder. Run the script/data_preprocessing.ipynb to process the data and split the dataset. The preprocessed data will be stored in the data/ directory.

### 3 Run the Model

Use the command:

```
python3 script/breaker_tf.py --suffix demo --file breaker_tf.py --gpu-index 0
```

Clear explanations for the command can be found within the code (see the parse_args function). You can define the model structure through the 'params' in the breaker_tf.py script.

### 4 Model Logging

The logs from the model run, parameter configurations, and checkpoint files are saved in the 'log' directory.

