import copy
import pandas as pd
from modeling.tuner import NetworkTuner

if __name__ == "__main__":

    # Define parameters to tune
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
        'use_early_stopping': [0, 1],
        'n_days': [1, 2, 3]
    }

    # Build the test dataframe
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

    # Define which columns are feature(s) and which are the target(s)
    X_cols = list(df.columns)
    y_cols = 'banana'

    # Instantiate our NetworkTuner
    nt = NetworkTuner(
        df=df, X_cols=X_cols,
        y_cols=y_cols, k_folds=5, max_n_days=3
    )

    nt.tune(
        'Albert', max_epochs=100
    )
