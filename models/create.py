# import os
# import sys
# import time
# import datetime
# import importlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.regularizers import l2, l1

# import kerastuner as kt
from keras import backend as K


class DummyHp():
    """A dummy class for hyperband, used when using parameters
    and not the lists for tuning"""
    def Choice(self, x, y):
        return y


class NetworkCreator():
    def __init__(self, df, X_cols, y_cols, n_days,
                 test_split=250, val_split=20):
        self.df = df
        self.X_cols = X_cols
        self.y_cols = y_cols
        self.test_split = test_split
        self.val_split = val_split
        self.prepare_data(n_days)

    def prepare_data(self, n_days=1):
        self.clean_cols()
        self.split_dataframes(self.test_split, self.val_split)
        self.split_and_scale_dataframes()
        self.reshape_data()

        self.create_TS_generators(n_days=n_days)
        self.input_shape = (self.X_train_reshaped.shape[0],
                            self.X_train_reshaped.shape[1])

    def build_and_fit_model(
        self,
        hp=None,

        # Input Layer
        input_neurons=64,
        input_dropout_rate=0,
        use_input_regularizer=0,
        input_regularizer_penalty=0,

        # Hidden layer
        n_hidden_layers=1,
        hidden_layer_activation='relu',
        hidden_dropout_rate=.3,
        hidden_neurons=64,
        use_hidden_regularizer=0,
        hidden_regularizer_penalty=0,

        # Early Stopping
        use_early_stopping=True,
        monitor='val_loss',
        patience=5,

        # Model fit
        epochs=2000,
        batch_size=32,
        shuffle=False,

        # Other
        dummy_hp=False
                            ):

        if not hp and dummy_hp:
            hp = DummyHp()
        elif not hp and not dummy_hp:
            string = "No hp implemented, did you want dummy_hp=True?"
            raise AttributeError(string)

        # Possible clear old session
        try:
            del self.model
            K.clear_session()
        except AttributeError:
            pass

        # Model creation
        self.model = Sequential()

        #   Input layer
        #       Regularizer check
        _reg = None
        _use_reg = hp.Choice('use_input_regularizer',
                             use_input_regularizer)
        if _use_reg:
            _penalty = hp.Choice('input_regularizer_penalty',
                                 input_regularizer_penalty)
            if _use_reg > 1:
                _reg = l2(_penalty)
            else:
                _reg = l1(_penalty)

        #       Add input layer
        self.model.add(LSTM(hp.Choice('input_neurons', input_neurons),
                            input_shape=self.input_shape,
                            kernel_regularizer=_reg))

        #           Dropout layer
        input_dropout_rate = hp.Choice('input_dropout_rate',
                                       input_dropout_rate)
        if input_dropout_rate != 0:
            self.model.add(Dropout(input_dropout_rate))

        #   Hidden layers
        #       Regularizer check
        _reg = None
        _use_reg = hp.Choice('use_hidden_regularizer',
                             use_hidden_regularizer)
        if _use_reg:
            _penalty = hp.Choice('hidden_regularizer_penalty',
                                 hidden_regularizer_penalty)
            if _use_reg > 1:
                _reg = l2(_penalty)
            else:
                _reg = l1(_penalty)

        #       Dropout check
        hidden_dropout_rate = hp.Choice('hidden_dropout_rate',
                                        hidden_dropout_rate)
        for i in range(hp.Choice('n_hidden_layers', n_hidden_layers)):
            self.model.add(
                Dense(hp.Choice('hidden_neurons',
                                hidden_neurons),
                      activation=hidden_layer_activation,
                      kernel_regularizer=_reg))

        #       Dropout layer
            if hidden_dropout_rate != 0:
                self.model.add(Dropout(hidden_dropout_rate))

        #   Output Layer
        self.model.add(Dense(len(self.y_cols)))

        #   Compile
        self.model.compile(optimizer='adam',
                           loss='mse')

        #   Define callbacks
        model_callbacks = []
        monitor = monitor
        patience = hp.Choice('patience', patience)
        if use_early_stopping:
            model_callbacks.append(EarlyStopping(monitor=monitor,
                                                 patience=patience))

        # Fit partial
        self.model.fit = partial(
            self.model.fit,
            callbacks=model_callbacks,
            # epochs=hp.Choice('epochs', epochs),
            batch_size=hp.Choice('batch_size', batch_size),
            shuffle=shuffle
            )
        return self.model

    # def run_test(self):
    #     early_stopping_kwargs = dict(monitor='val_loss', patience=25)
    #     model_fit_kwargs = dict(epochs=2000, batch_size=64,
    #                             verbose=2, shuffle=False)
    #     self.build_and_fit_model(early_stopping_kwargs=early_stopping_kwargs,
    #                              model_fit_kwargs=model_fit_kwargs)

    #     self.predict_r2_scores()
    #     self.display_r2_scores()

    def clean_cols(self):
        """
        If a string is given for cols then it will take all of the columns
        that are in the dataframe that contain that string
        """
        if isinstance(self.X_cols, str):
            self.X_cols = \
                [col for col in self.df.columns if self.X_cols in col]
            print("Got", len(self.X_cols), "X columns")

        if isinstance(self.y_cols, str):
            self.y_cols = \
                [col for col in self.df.columns if self.y_cols in col]
            print("Got", len(self.y_cols), "y columns")

    def _contains(self, sub, pri):
        "https://stackoverflow.com/questions/3847386/how-to-test-if-a-list-contains-another-list"  # noqa
        M, N = len(pri), len(sub)
        i, LAST = 0, M-N+1
        while True:
            try:
                found = pri.index(sub[0], i, LAST)  # find first elem in sub
            except ValueError:
                return False
            if pri[found:found+N] == sub:
                return [found, found+N-1]
            else:
                i = found+1

    def split_dataframes(self, test_split=250, val_split=20):
        """
        Splits the dataframe into train, test, and validation
        """
        if self.X_cols == self.y_cols:
            print('y is the same as x')
            self.df = self.df[self.X_cols]
        elif ((self._contains(self.y_cols, self.X_cols)[1]
              - self._contains(self.y_cols, self.X_cols)[0])
              + 1 == len(self.y_cols)):
            print("y is in x")
        else:
            print('y is different than x')
            select_cols = self.X_cols + self.y_cols
            self.df = self.df[select_cols]

        # Get column indices
        self.column_indices = \
            {name: i for i, name in enumerate(self.df.columns)}

        # Split dataframes
        self.df_train = self.df.iloc[:test_split].copy()
        self.df_test = self.df.iloc[test_split:-val_split].copy()
        self.df_val = self.df.iloc[-val_split::].copy()

    def split_and_scale_dataframes(self):
        """
        Scales and splits the data into X and y of each
        train, test, and val.
        """
        if self.X_cols != self.y_cols:  # Accounting for another scaler
            # Split df by X or y cols
            self.X_df_train = self.df_train[self.X_cols]
            self.y_df_train = self.df_train[self.y_cols]
            self.X_df_test = self.df_test[self.X_cols]
            self.y_df_test = self.df_test[self.y_cols]
            self.X_df_val = self.df_val[self.X_cols]
            self.y_df_val = self.df_val[self.y_cols]

            # Define scalers
            self.X_scaler = MinMaxScaler()
            self.y_scaler = MinMaxScaler()

            # Scale data
            self.X_train = self.X_scaler.fit_transform(self.X_df_train)
            self.y_train = self.y_scaler.fit_transform(self.y_df_train)
            self.X_test = self.X_scaler.transform(self.X_df_test)
            self.y_test = self.y_scaler.transform(self.y_df_test)
            self.X_val = self.X_scaler.transform(self.X_df_val)
            self.y_val = self.y_scaler.transform(self.y_df_val)

        else:  # If X and y are the same i.e predicting self with self

            # Define scaler, y is 0 because it is not used
            self.X_scaler = MinMaxScaler()
            self.y_scaler = 0

            # Scale data
            self.df_train_scaled = self.X_scaler.fit_transform(self.df_train)
            self.df_test_scaled = self.X_scaler.transform(self.df_test)
            self.df_val_scaled = self.X_scaler.transform(self.df_val)

            # Split data
            self.X_train = self.df_train_scaled.copy()
            self.y_train = self.df_train_scaled.copy()
            self.X_test = self.df_test_scaled.copy()
            self.y_test = self.df_test_scaled.copy()
            self.X_val = self.df_val_scaled.copy()
            self.y_val = self.df_val_scaled.copy()

    def reshape_data(self):
        """
        Reshapes the data based on X_train, and y_train's shapes
        """
        # Get n_features
        self.X_n_features = self.X_train.shape[1]
        #print(self.X_train.shape)
        self.y_n_features = self.y_train.shape[1]
        #print(self.y_train.shape)

        # Reshape data
        self.X_train_reshaped = self.X_train.reshape((len(self.X_train),
                                                        self.X_n_features))
        self.y_train_reshaped = self.y_train.reshape((len(self.y_train),
                                                        self.y_n_features))

        self.X_test_reshaped = self.X_test.reshape((len(self.X_test),
                                                    self.X_n_features))
        self.y_test_reshaped = self.y_test.reshape((len(self.y_test),
                                                    self.y_n_features))

        self.X_val_reshaped = self.X_val.reshape((len(self.X_val),
                                                    self.X_n_features))
        self.y_val_reshaped = self.y_val.reshape((len(self.y_val),
                                                    self.y_n_features))

    def create_TS_generators(self, n_days):
        """
        Creates the data generators
        """
        self.n_input = n_days
        # Get data generators
        self.train_data_gen = sequence.TimeseriesGenerator(
                            self.X_train_reshaped,
                            self.y_train_reshaped,  # [:,column_indices[self.y_cols]],
                            length=self.n_input)
        self.test_data_gen = sequence.TimeseriesGenerator(
                            self.X_test_reshaped,
                            self.y_test_reshaped,  # [:,column_indices[self.y_cols]],
                            length=self.n_input)
        self.val_data_gen = sequence.TimeseriesGenerator(
                            self.X_val_reshaped,
                            self.y_val_reshaped,  # [:,column_indices[self.y_cols]],
                            length=self.n_input)

    def get_r2_scores_one_y(self, plot=False):
        cols = self.y_cols + ['predicted']

        # Training data
        train_prediction = self.model.predict(self.train_data_gen)
        train_prediction_iv = self.y_scaler.inverse_transform(train_prediction)
        train_prediction_idx = self.df_train[self.n_input:][self.y_cols].index
        self.df_train['predicted'] = pd.DataFrame(train_prediction_iv,
                                                  columns=self.y_cols,
                                                  index=train_prediction_idx)
        self.df_predict_train = self.df_train[cols].dropna()
        self.train_r2 = r2_score(self.df_predict_train[self.y_cols],
                                 self.df_predict_train[['predicted']])

        # Testing data
        test_prediction = self.model.predict(self.test_data_gen)
        test_prediction_iv = self.y_scaler.inverse_transform(test_prediction)
        test_prediction_idx = self.df_test[self.n_input:][self.y_cols].index
        self.df_test['predicted'] = pd.DataFrame(test_prediction_iv,
                                                 columns=self.y_cols,
                                                 index=test_prediction_idx)
        self.df_predict_test = self.df_test[cols].dropna()
        self.test_r2 = r2_score(self.df_predict_test[self.y_cols],
                                self.df_predict_test[['predicted']])

        # Validation data
        val_prediction = self.model.predict(self.val_data_gen)
        val_prediction_iv = self.y_scaler.inverse_transform(val_prediction)
        val_prediction_idx = self.df_val[self.n_input:][self.y_cols].index
        self.df_val['predicted'] = pd.DataFrame(val_prediction_iv,
                                                columns=self.y_cols,
                                                index=val_prediction_idx)
        self.df_predict_val = self.df_val[cols].dropna()
        self.val_r2 = r2_score(self.df_predict_val[self.y_cols],
                               self.df_predict_val[['predicted']])

        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 6))
            self.df_predict_train[cols].plot(ax=ax1)
            self.df_predict_test[cols].plot(ax=ax2)
            self.df_predict_val[cols].plot(ax=ax3)

    def get_r2_scores_multi_y(self):
        # Training data
        train_prediction = self.model.predict(self.train_data_gen)
        train_prediction_iv = self.y_scaler.inverse_transform(train_prediction)
        train_prediction_idx = self.y_df_train[self.n_input:].index
        self.df_predict_train = pd.DataFrame(train_prediction_iv,
                                             columns=self.y_cols,
                                             index=train_prediction_idx)
        y_true_train = self.y_df_train.loc[self.df_predict_train.index]
        self.train_r2 = r2_score(y_true_train, self.df_predict_train)

        # Testing data
        test_prediction = self.model.predict(self.test_data_gen)
        test_prediction_iv = self.y_scaler.inverse_transform(test_prediction)
        test_prediction_idx = self.y_df_test[self.n_input:].index
        self.df_predict_test = pd.DataFrame(test_prediction_iv,
                                            columns=self.y_cols,
                                            index=test_prediction_idx)
        y_true_test = self.y_df_test.loc[self.df_predict_test.index]
        self.test_r2 = r2_score(y_true_test, self.df_predict_test)

        # Validation data
        val_prediction = self.model.predict(self.val_data_gen)
        val_prediction_iv = self.y_scaler.inverse_transform(val_prediction)
        val_prediction_idx = self.y_df_val[self.n_input:].index
        self.df_predict_val = pd.DataFrame(val_prediction_iv,
                                           columns=self.y_cols,
                                           index=val_prediction_idx)
        y_true_val = self.y_df_val.loc[self.df_predict_val.index]
        self.val_r2 = r2_score(y_true_val, self.df_predict_val)

    def get_r2_scores_self_y(self):
        # Training data
        train_prediction = self.model.predict(self.train_data_gen)
        train_prediction_iv = self.X_scaler.inverse_transform(train_prediction)
        train_prediction_idx = self.df_train[self.n_input:].index
        self.df_predict_train = pd.DataFrame(train_prediction_iv,
                                             columns=self.y_cols,
                                             index=train_prediction_idx)
        y_true_train = self.df_train.loc[self.df_predict_train.index]
        self.train_r2 = r2_score(y_true_train, self.df_predict_train)

        # Testing data
        test_prediction = self.model.predict(self.test_data_gen)
        test_prediction_iv = self.X_scaler.inverse_transform(test_prediction)
        test_prediction_idx = self.df_test[self.n_input:].index
        self.df_predict_test = pd.DataFrame(test_prediction_iv,
                                            columns=self.y_cols,
                                            index=test_prediction_idx)
        y_true_test = self.df_test.loc[self.df_predict_test.index]
        self.test_r2 = r2_score(y_true_test, self.df_predict_test)

        # Validation data
        val_prediction = self.model.predict(self.val_data_gen)
        val_prediction_iv = self.X_scaler.inverse_transform(val_prediction)
        val_prediction_idx = self.df_val[self.n_input:].index
        self.df_predict_val = pd.DataFrame(val_prediction_iv,
                                           columns=self.y_cols,
                                           index=val_prediction_idx)
        y_true_val = self.df_val.loc[self.df_predict_val.index]
        self.val_r2 = r2_score(y_true_val, self.df_predict_val)

    def predict_r2_scores(self, plot=False):
        # Predictions (r2_score)
        # If predicting one column for y
        if self.y_scaler and (len(self.y_cols) == 1):
            self.get_r2_scores_one_y(plot)

        # If predicting multiple columns for y
        elif self.y_scaler:
            self.get_r2_scores_multi_y()

        # if predicting self with self
        else:
            self.get_r2_scores_self_y()

    def display_r2_scores(self):
        print("-"*20, "\nr2 scores\n" + "-"*20)
        print(f"Training:{self.train_r2:.2f}")
        print(f"Testing:{self.test_r2:.2f}")
        print(f"Validation:{self.val_r2:.2f}")


if __name__ == "__main__":
    USE_GPU = True
    IGNORE_WARN = True
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

    df = pd.read_pickle('./data/modeling/model_df.pkl')
    # self.X_cols = [col for col in self.df.columns if 'TSLA_price' not in col]
    # self.y_cols = self.X_cols
    X_cols = [col for col in df.columns if 'TSLA' in col]
    y_cols = X_cols
    n_days = 1

    creator = NetworkCreator(df, X_cols, y_cols)
    creator.run_test()
