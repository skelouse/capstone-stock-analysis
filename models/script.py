import os
import shutil
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
 - narcissus
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
    'input_neurons': [2, 4, 8, 16],
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


class Logger():
    x = 0

    @classmethod
    def clear_checkpoints(cls):
        print("Clearing checkpoints")
        project_files = os.listdir(cls.filepath)
        for file in project_files:
            try:
                shutil.rmtree(cls.filepath+file+'/checkpoints')
            except FileNotFoundError:
                pass

    @classmethod
    def report_trial_state(cls, *args):
        cls.x += 1
        print("trial", cls.x)
        if cls.x >= 5:
            cls.clear_checkpoints()
            cls.x = 0

    @classmethod
    def register_directory(cls, name):
        cls.filepath = f'./tuner_directory/{name}/'

    def register_tuner(*args):
        pass

    def register_trial(*args):
        pass



def run(df, X_cols, y_cols, n_days, name, max_epochs):
    creator = NetworkCreator(df, X_cols, y_cols, n_days,
                             test_split=300, val_split=False)
    creator.build_and_fit_model = partial(
        creator.build_and_fit_model, **parameters
    )
    Logger.register_directory(name)
    tuner = kt.Hyperband(creator.build_and_fit_model,
                         objective='val_loss',
                         max_epochs=max_epochs,
                         factor=3,
                         directory='./tuner_directory',
                         project_name=name,
                         logger=Logger)

    tuner.search(creator.train_data_gen,
                 validation_data=(creator.test_data_gen))

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete.
    The optimal number of units in the first densely-connected layer
    {best_hps.__dict__['values']}
    """)


# HERMES
def hermes(n_days, max_epochs):
    # Hermes is company and performance
    df_hermes = pd.read_pickle("./data/modeling/hermes.pkl")
    df_analyst = pd.read_pickle("./data/modeling/analyst.pkl")
    X_cols = [col for col in df_hermes.columns
              if col not in df_analyst.columns]
    y_cols = [col for col in df_hermes.columns
              if col in df_analyst.columns]
    run(df_hermes, X_cols, y_cols, n_days, 'Hermes', max_epochs)


# NARCISSUS
def narcissus(n_days, max_epochs):
    df_performance = pd.read_pickle("./data/modeling/performance.pkl")
    X_cols = list(df_performance.columns)
    y_cols = X_cols
    run(df_performance, X_cols, y_cols, n_days, 'Narcissus', max_epochs)


# CRONUS
def cronus(n_days, max_epochs):
    df_analyst = pd.read_pickle("./data/modeling/analyst.pkl")
    df_prices = pd.read_pickle("./data/modeling/prices.pkl")

    df_cronus = pd.concat([df_analyst, df_prices], axis=1)
    X_cols = list(df_cronus.columns)
    y_cols = list(df_prices.columns)
    run(df_cronus, X_cols, y_cols, n_days, 'Cronus', max_epochs)


# Combined model
def all(n_days, max_epochs):
    model_df = pd.read_pickle("./data/modeling/model_df.pkl")
    X_cols = list(model_df.columns)
    y_cols = list(model_df.columns)
    run(model_df, X_cols, y_cols, n_days, 'All', max_epochs)

def sym(n_days, max_epochs, symbol):
    model_df = pd.read_pickle("./data/modeling/model_df.pkl")
    X_cols = list(model_df.columns)
    y_cols = f'{symbol}_price'
    run(model_df, X_cols, y_cols, n_days, symbol.lower(), max_epochs)

if __name__ == "__main__":
    models = sys.argv[1::]
    if isinstance(models, str):
        models = list(models)
    print('TRAINING: ', models)
    df_prices = pd.read_pickle("./data/modeling/prices.pkl")
    all_symbols = pd.read_pickle('./data/prices.pkl').reset_index() \
        .set_index('date', drop=True)['sym'].unique()
    newest_day = df_prices.iloc[[-1]].index[0].strftime("%Y-%m-%d")
    input(f"\n\nData last updated {newest_day} @ 8PM, okay? ( press enter )\n\n")
    max_epochs = 5000
    # Query here for n_days and max_epochs

    for model in models:
        print(f"TRAINING {model}!")
        print(f"max_epochs -> {max_epochs}")
        print("\nStarting in 10 seconds")
        print("")
        with open("current.txt", 'w') as f:
            f.write(model)
        # for i in list(range(10))[::-1]:
        #     time.sleep(1)
        #     print(f'  {i}', end='\r')
        if model == 'hermes':
            hermes(3, max_epochs)
        elif model == 'cronus':
            cronus(43, max_epochs)
        elif model == 'narcissus':
            narcissus(43, max_epochs)
        elif model == 'all':
            all(4, max_epochs)
        elif model.upper() in all_symbols:
            sym(4, max_epochs, model.upper())



# filepath = './tuner_directory/aapl/'

# project_files = os.listdir(filepath)
# for file in project_files:
#     try:
#         shutil.rmtree(filepath+file+'/checkpoints')
#     except FileNotFoundError:
#         pass