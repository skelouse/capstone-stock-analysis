import copy
import pandas as pd
from modeling.tuner import NetworkTuner


def train_aapl_all_sectors():
    model_df = pd.read_pickle("./data/modeling/model_df_cols_dropped.pkl")
    company_df = pd.read_pickle("./data/company.pkl")
    industry = company_df.loc['AAPL']['industry']
    industry_syms = list(company_df.loc[
                         company_df['industry'] == industry].index)
    slice_cols = [col for col in model_df.columns
                  if col.split('_')[0] in industry_syms]
    industry_df = model_df[slice_cols].copy()

    X_cols = list(industry_df.columns)
    y_cols = 'price'

    nt = NetworkTuner(
        df=industry_df, X_cols=X_cols,
        y_cols=y_cols, k_folds=5, max_n_days=8
    )
    batch_size = nt.X_n_features
    parameters = {
        'input_dropout_rate': [.1, .3, .5],
        'use_input_regularizer': [0, 1, 2],
        'input_regularizer_penalty': [0.01, 0.1],  # 0.01, 0.05, 0.1, 0.3
        'add_hidden_lstm': [0, 1],
        'hidden_lstm_neurons': [32, 64],
        'add_gaussian_noise': [0, 1],
        'gaussian_noise_quotient': [.5, 1.0, 3.0],
        'n_hidden_layers': [1, 2, 4],
        'hidden_dropout_rate': [0.0, .1, .3],  # .3, .5, .9
        'hidden_neurons': [16, 32, 64, batch_size],
        'use_hidden_regularizer': [0, 1, 2],
        'hidden_regularizer_penalty': [0.01, 0.1],  # 0.01, 0.05, 0.1, 0.3
        'patience': [0],  # [5, 25, 50, 100],
        'batch_size': [64, 128, batch_size],
        'use_early_stopping': [0],  # [0, 1]
        'n_days': [1, 2, 3, 8],
        'optimizer': ['adam', 'rmsprop']
    }

    nt.tune(
        'aapl_industry', 2000, **parameters
    )

def test():

    batch_size = 1
    # Define parameters to tune
    parameters = {
        'input_dropout_rate': [.1, .3, .5],
        'use_input_regularizer': [0, 1, 2],
        'input_regularizer_penalty': [0.01, 0.1],  # 0.01, 0.05, 0.1, 0.3
        'add_hidden_lstm': [0, 1],
        'hidden_lstm_neurons': [32, 64],
        'add_gaussian_noise': [0, 1],
        'gaussian_noise_quotient': [.5, 1.0, 3.0],
        'n_hidden_layers': [0, 1, 2, 4],
        'hidden_dropout_rate': [0.0, .1, .3],  # .3, .5, .9
        'hidden_neurons': [16, 32, 64],
        'use_hidden_regularizer': [0, 1, 2],
        'hidden_regularizer_penalty': [0.01, 0.1],  # 0.01, 0.05, 0.1, 0.3
        'patience': [0],  # [5, 25, 50, 100],
        'batch_size': [batch_size],
        'use_early_stopping': [0],  # [0, 1]
        'n_days': [2],
        'optimizer': ['adam', 'rmsprop']
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
    y_cols = ['banana', 'orange']

    # Instantiate our NetworkTuner
    nt = NetworkTuner(
        df=df, X_cols=X_cols,
        y_cols=y_cols, k_folds=5, max_n_days=4
    )
    nt.tune(
        'Albert', 100, **parameters
    )


if __name__ == "__main__":
    train_aapl_all_sectors()
    # test()