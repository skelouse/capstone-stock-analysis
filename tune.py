import os
import shutil
import pandas as pd
from functools import partial
import copy
import kerastuner as kt

from modeling.tuner import NetworkTuner
from modeling.create import NetworkCreator

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


def run(name, max_epochs=10, *args, **kwargs):
    nt = NetworkTuner(*args, **kwargs)
    nt.build_and_fit_model = partial(
        nt.build_and_fit_model, **parameters
    )
    Logger.register_directory(name)
    tuner = kt.Hyperband(nt.build_and_fit_model,
                         objective='val_loss',
                         max_epochs=max_epochs,
                         factor=3,
                         directory='./tuner_directory',
                         project_name=name,
                         logger=Logger)

    tuner.search(nt)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete.
    The optimal number of units in the first densely-connected layer
    {best_hps.__dict__['values']}
    """)

if __name__ == "__main__":
    _list = list(range(20))
    df = pd.DataFrame({
        'apple': copy.copy(_list),
        'orange': copy.copy(_list),
        'banana': copy.copy(_list),
        'pear': copy.copy(_list),
        'cucumber': copy.copy(_list),
        'tomato': copy.copy(_list),
        'plum': copy.copy(_list),
        'watermelon': copy.copy(_list)
    })

    X_cols = list(df.columns)
    y_cols = 'banana'

    run('Albert', df=df, X_cols=X_cols, y_cols=y_cols,
        max_n_days=4, k_folds=3, max_epochs=100)
