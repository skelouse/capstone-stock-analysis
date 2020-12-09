import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
import kerastuner as kt
from create import NetworkCreator
import sys
script = """
--help
sequence of models i.e 'python script model1 model2'
Available models:
 - hermes
 - cronus
 - narcissus (Partially trained)
"""
try:
    if sys.argv[1] == "--help":
        print(script)
        exit()
except IndexError:
    print("""
          ERROR: (No models specified)
          run script with --help for information
          """)
    exit()

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

parameters = {
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


def run(df, X_cols, y_cols, n_days, name):
    creator = NetworkCreator(df, X_cols, y_cols, n_days,
                             test_split=300, val_split=1)

    creator.build_and_fit_model = partial(
        creator.build_and_fit_model, **parameters
    )
    tuner = kt.Hyperband(creator.build_and_fit_model,
                         objective='val_loss',
                         max_epochs=500,
                         factor=3,
                         directory='./tuner_directory',
                         project_name=name)

    tuner.search(creator.train_data_gen,
                 validation_data=(creator.test_data_gen))

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete.
    The optimal number of units in the first densely-connected layer
    {best_hps.__dict__['values']}
    """)


# HERMES
def hermes():
    # Hermes is company and performance
    df_hermes = pd.read_pickle("./data/modeling/hermes.pkl")
    df_analyst = pd.read_pickle("./data/modeling/analyst.pkl")
    X_cols = [col for col in df_hermes.columns
              if col not in df_analyst.columns]
    y_cols = [col for col in df_hermes.columns
              if col in df_analyst.columns]
    n_days = 44
    run(df_hermes, X_cols, y_cols, n_days, 'Hermes')


# NARCISSUS
def narcissus():
    df_performance = pd.read_pickle("./data/modeling/performance.pkl")
    X_cols = list(df_performance.columns)
    y_cols = X_cols
    n_days = 44
    run(df_performance, X_cols, y_cols, n_days, 'Narcissus')


# CRONUS
def cronus():
    df_analyst = pd.read_pickle("./data/modeling/analyst.pkl")
    df_prices = pd.read_pickle("./data/modeling/prices.pkl")

    df_cronus = pd.concat([df_analyst, df_prices], axis=1)
    X_cols = list(df_cronus.columns)
    y_cols = list(df_prices.columns)
    # n_days = 15 # .59
    n_days = 44
    run(df_cronus, X_cols, y_cols, n_days, 'Cronus')


if __name__ == "__main__":
    models = sys.argv[1::]
    if isinstance(models, str):
        models = list(models)
    print('TRAINING: ', models)
    print("Starting in 10 seconds")
    print("Have you update me and my data?")
    print("")
    for i in list(range(10))[::-1]:
        time.sleep(1)
        print(i, end='\r')
    for model in models:
        if model == 'hermes':
            hermes()
        elif model == 'cronus':
            cronus()
        elif model == 'narcissus':
            narcissus()
