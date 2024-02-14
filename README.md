# Breaker: Removing Shortcut Cues with User Clustering for Single-slot Recommendation System

This repo contains the implementation of Breaker in Tensorflow and PyTorch.

## Directory Structure Explanation

```shell
├── README.md
├── config # directory for configuration files related to data preprocessing
│   └── feat_info.json
├── data # directory for data files
│   ├── demo_data.csv
│   ├── test_data_fornn.csv
│   └── train_data_fornn.csv
├── log # directory for run logs
├── requirements.txt # list of dependencies for the project environment
└── script
    ├── breaker_tf.py  # script implemented with TensorFlow
    ├── breaker_torch # directory with Torch implementation scripts
    └── data_preprocessing.ipynb # Jupyter notebook for data preprocessing
```

## How to run? 

### 1 Prepare the Environment

```python
# Requires Python 3.6.15
pip install -r requirements.txt
```

### 2 Data Preprocessing

You will need to prepare a configuration file for data preprocessing, as detailed in `feat_info.json`. This configuration file should include three fields:

- `colType`: Specifies the type of each feature, with 'E' indicating a categorical feature and 'M' indicating a continuous feature.
- `encode`: For categorical features ('E'), provides a dictionary that maps original feature values to their encoded values, which will be used for embedding purposes. Ensure each categorical feature has its corresponding encoding dictionary.
- `manualBox`: For continuous features ('M'), this indicates the split points for converting the feature values into discrete intervals. Split points should be specified as a list of values.

An example dataset, `demo_data.csv`, is available in the `data/` folder. To process the data and split the dataset, open and run the Jupyter notebook located at `script/data_preprocessing.ipynb`. The preprocessed data will be saved in the `data/` directory with a specified file name and format, as described in the notebook.

### 3 Run the Model

Run the command:

```
python3 script/breaker_tf.py --suffix demo --file script/breaker_tf.py --gpu-index 0
```

Clear explanations for the command-line arguments are provided within the `parse_args` function in the script file. Please refer to this function for detailed descriptions of each argument. You can define the model structure by setting the appropriate parameters within the 'params' dictionary in the `breaker_tf.py` script.

### 4 Model Logging

All logs, including training progress, and relevant model run information, along with the parameter configurations set for the run, and the model's checkpoint files, are saved in the `log` directory.

