

import os
import sys
import time
import datetime
import importlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
# from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.preprocessing import sequence


class NetworkCreator():
    def __init__(self, df, X_cols, y_cols,
                 test_split=250, val_split=20):
        self.df = df
        self.X_cols = X_cols
        self.y_cols = y_cols
        self.test_split = test_split
        self.val_split = val_split

    def run_test(self):
        self.clean_cols()
        self.split_dataframes(self.test_split, self.val_split)
        self.split_and_scale_dataframes()
        self.reshape_data()

        self.create_TS_generators(n_days=4)
        self.input_shape = (self.X_train_reshaped.shape[0],
                            self.X_train_reshaped.shape[1])

        # Temp
        early_stopping_kwargs = dict(monitor='val_loss', patience=25)
        model_fit_kwargs = dict(epochs=2000, batch_size=64,
                                verbose=2, shuffle=False)
        self.model(early_stopping_kwargs=early_stopping_kwargs,
                   model_fit_kwargs=model_fit_kwargs)

        self.predict_r2_scores()
        self.display_r2_scores()

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

    def split_dataframes(self, test_split=250, val_split=20):
        """
        Splits the dataframe into train, test, and validation
        """
        if self.X_cols == self.y_cols:
            print('y is the same as x')
            self.df = self.df[self.X_cols]
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
        self.y_n_features = self.y_train.shape[1]

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
        print("Training:", self.train_r2)
        print("Testing:", self.test_r2)
        print("Validation:", self.val_r2)

    def model(self,
              input_neurons=64,
              input_dropout_rate=0,
              input_kwargs={},
              input_dropout_kwargs={},
              optimizer='adam',
              loss='mse',
              n_hidden_layers=1,
              hidden_layer_activation='relu',
              hidden_dropout_rate=.3,
              hidden_neurons=64,
              hidden_kwargs={},
              hidden_dropout_kwargs={},
              output_layer_kwargs={},
              use_early_stopping=True,
              early_stopping_kwargs={},
              model_compile_kwargs={},
              model_fit_kwargs={},
              ):

        # Model creation
        self.model = Sequential()

        # Input layer
        self.model.add(LSTM(input_neurons,
                            input_shape=self.input_shape,
                            **input_kwargs))
        if input_dropout_rate != 0:
            self.model.add(Dropout(input_dropout_rate,
                                   **input_dropout_kwargs))

        # Hidden layers
        for i in range(n_hidden_layers):
            self.model.add(Dense(hidden_neurons,
                                 activation=hidden_layer_activation,
                                 **hidden_kwargs))
            if hidden_dropout_rate != 0:
                self.model.add(Dropout(hidden_dropout_rate,
                                       **hidden_dropout_kwargs))

        # Output Layer
        self.model.add(Dense(len(self.y_cols),
                             **output_layer_kwargs))

        # Compile
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           **model_compile_kwargs)

        # Define callbacks
        model_callbacks = []
        if use_early_stopping:
            model_callbacks.append(EarlyStopping(**early_stopping_kwargs))

        # Fit
        self.history = self.model.fit(self.train_data_gen,
                                      validation_data=(self.test_data_gen),
                                      callbacks=model_callbacks,
                                      **model_fit_kwargs)


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



