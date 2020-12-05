import os
import sys
import time
import datetime
import importlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
# from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.preprocessing import sequence

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# Enable GPU

import kerastuner as kt

from create import NetworkCreator

USE_GPU = True
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
if USE_GPU:
    # Enable GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0],
                                             enable=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Show GPU
    print("Using GPU")
    print(tf.config.list_physical_devices('GPU'))
else:
    print("Using CPU")


# df_company = pd.read_pickle("./data/modeling/company.pkl")

# NARCISSUS
# df_performance = pd.read_pickle("./data/modeling/performance.pkl")
# X_cols = list(df_performance.columns)
# y_cols = X_cols
# n_days = 4
# creator = NetworkCreator(df_performance, X_cols, y_cols, n_days)

# # Best NARCISSUS
# values= {
#     "use_input_regularizer": 2,
#     "input_neurons": 64,
#     "input_dropout_rate": 0.1,
#     "use_hidden_regularizer": 0,
#     "hidden_dropout_rate": 0,
#     "n_hidden_layers": 3,
#     "hidden_neurons": 64,
#     "patience": 5,
#     "batch_size": 128,
#     "input_regularizer_penalty": 0.01,
#     "hidden_regularizer_penalty": 0.01,
#     "tuner/epochs": 4,
#     "tuner/initial_epoch": 0,
#     "tuner/bracket": 4,
#     "tuner/round": 0
# }


# CRONUS

df_analyst = pd.read_pickle("./data/modeling/analyst.pkl")
df_prices = pd.read_pickle("./data/modeling/prices.pkl")

df_cronus = pd.concat([df_analyst, df_prices], axis=1)
X_cols = list(df_cronus.columns)
y_cols = list(df_prices.columns)
n_days = 4
creator = NetworkCreator(df_cronus, X_cols, y_cols, n_days)




parameters = {
    # 'epochs': list(range(100, 2000, 50)),
    'input_neurons': [16, 32, 64],
    'input_dropout_rate': [.1, .3, .5],
    'use_input_regularizer': [0, 1, 2],
    'input_regularizer_penalty': [0.01, 0.05, 0.1, 0.3],
    'n_hidden_layers': [1, 3, 5, 8],
    'hidden_dropout_rate': [0.0, .3, .5, .9],
    'hidden_neurons': [16, 32, 64],
    'use_hidden_regularizer': [0, 1, 2],
    'hidden_regularizer_penalty': [0.01, 0.05, 0.1, 0.3],
    'patience': [5, 25, 50, 100],
    'batch_size': [32, 64, 128],
    'use_early_stopping': [0, 1]
}
creator.build_and_fit_model = partial(
    creator.build_and_fit_model, **parameters
)
tuner = kt.Hyperband(creator.build_and_fit_model,
                     objective='val_loss',
                     max_epochs=300,
                     factor=3,
                     directory='./tuner_directory',
                     project_name='Narcissus')

tuner.search(creator.train_data_gen, validation_data=(creator.test_data_gen))


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete.
The optimal number of units in the first densely-connected layer
is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
