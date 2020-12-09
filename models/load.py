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

import kerastuner as kt
from keras import backend as K

from .create import NetworkCreator


class NetworkLoader():
    def __init__(self, df, X_cols, y_cols, model_name,
                 n_days=4, split=20,
                 tuner_directory='E:/capstone/Tuners'):
        if model_name.title() not in ['Hermes', 'Cronus', 'Narcissus']:
            raise NameError("%s is not a valid model name"
                            % model_name.title())
        self.df = df
        self.X_cols = X_cols
        self.y_cols = y_cols
        self.n_days = n_days
        self.tuner_directory = tuner_directory
        self.test_split = split
        self.prepare_data(model_name.title())

    def prepare_data(self, name):
        self.clean_cols()
        self.split_dataframes(self.test_split)
        self.split_and_scale_dataframes()
        self.reshape_data()

        self.create_TS_generators(n_days=self.n_days)
        self.input_shape = (self.X_train_reshaped.shape[0],
                            self.X_train_reshaped.shape[1])
        #self.parameters = self.load_parameters(name)

    def load_parameters(self, name):
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
        self.creator = NetworkCreator(
            self.df,
            self.X_cols,
            self.y_cols,
            self.n_days)
        print("Making creator")
        self.creator.build_and_fit_model = partial(
            self.creator.build_and_fit_model, **parameters
                                             )
        tuner = kt.Hyperband(self.creator.build_and_fit_model,
                             objective='val_loss',
                             max_epochs=5000,
                             directory=self.tuner_directory,
                             project_name=name.title())

        # tuner.search(self.creator.train_data_gen,
        #              validation_data=(self.creator.test_data_gen))
        print("Getting top_10_hps")
        self.top_10_hps = tuner.get_best_hyperparameters(num_trials=10)
         # .__dict__['values'])

    def predict_n_days(self, n_days):
        pass

    def choice_wrap(self, x, y):
        return y

    def build_model(
        self,

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
                   ):

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
        _use_reg = self.choice_wrap('use_input_regularizer',
                                    use_input_regularizer)
        if _use_reg:
            _penalty = self.choice_wrap('input_regularizer_penalty',
                                        input_regularizer_penalty)
            if _use_reg > 1:
                _reg = l2(_penalty)
            else:
                _reg = l1(_penalty)

        #       Add input layer
        self.model.add(LSTM(self.choice_wrap('input_neurons', input_neurons),
                            input_shape=self.input_shape,
                            kernel_regularizer=_reg))

        #           Dropout layer
        input_dropout_rate = self.choice_wrap('input_dropout_rate',
                                              input_dropout_rate)
        if input_dropout_rate != 0:
            self.model.add(Dropout(input_dropout_rate))

        #   Hidden layers
        #       Regularizer check
        _reg = None
        _use_reg = self.choice_wrap('use_hidden_regularizer',
                                    use_hidden_regularizer)
        if _use_reg:
            _penalty = self.choice_wrap('hidden_regularizer_penalty',
                                        hidden_regularizer_penalty)
            if _use_reg > 1:
                _reg = l2(_penalty)
            else:
                _reg = l1(_penalty)

        #       Dropout check
        hidden_dropout_rate = self.choice_wrap('hidden_dropout_rate',
                                               hidden_dropout_rate)
        for i in range(self.choice_wrap('n_hidden_layers', n_hidden_layers)):
            self.model.add(
                Dense(self.choice_wrap('hidden_neurons',
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
        patience = self.choice_wrap('patience', patience)
        use_early_stopping = self.choice_wrap('use_early_stopping',
                                              use_early_stopping)
        if use_early_stopping:
            model_callbacks.append(EarlyStopping(monitor=monitor,
                                                 patience=patience))

        # Fit partial
        self.model.fit = partial(
            self.model.fit,
            callbacks=model_callbacks,
            epochs=self.choice_wrap('epochs', epochs),
            batch_size=self.choice_wrap('batch_size', batch_size),
            shuffle=shuffle
            )
        return self.model

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

    def split_dataframes(self, test_split):
        """
        Splits the data on columns
        """
        try:
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
        except TypeError:
            print('y is in dataframe but not x')
        self.df_train = self.df.iloc[:-test_split].copy()
        self.df_test = self.df.iloc[-test_split:].copy()


    def split_and_scale_dataframes(self):
        """
        Scales and splits the data into X and y of each
        train, test, and val.
        """
        if self.X_cols != self.y_cols:  # Accounting for another scaler
            # Split df by X or y cols
            self.X_df = self.df[self.X_cols]
            self.y_df = self.df[self.y_cols]

            self.X_df_train = self.df_train[self.X_cols]
            self.y_df_train = self.df_train[self.y_cols]
            self.X_df_test = self.df_test[self.X_cols]
            self.y_df_test = self.df_test[self.y_cols]

            # Define scalers
            self.X_scaler = MinMaxScaler()
            self.y_scaler = MinMaxScaler()

            # Scale data
            self.X = self.X_scaler.fit_transform(self.X_df)
            self.y = self.y_scaler.fit_transform(self.y_df)


            # self.X_train = self.X_scaler.fit_transform(self.X_df_train)
            # self.y_train = self.y_scaler.fit_transform(self.y_df_train)
            self.X_train = self.X_scaler.transform(self.X_df_train)
            self.y_train = self.y_scaler.transform(self.y_df_train)
            self.X_test = self.X_scaler.transform(self.X_df_test)
            self.y_test = self.y_scaler.transform(self.y_df_test)
            # self.X_train = (self.X_df_train).values
            # self.y_train = (self.y_df_train).values
            # self.X_test = (self.X_df_test).values
            # self.y_test = (self.y_df_test).values

        else:  # If X and y are the same i.e predicting self with self

            # Define scaler, y is 0 because it is not used
            self.X_scaler = MinMaxScaler()
            self.y_scaler = 0

            # Scale data
            self.df_scaled = self.X_scaler.fit_transform(self.df)
            self.df_train_scaled = self.X_scaler.transform(self.df_train)
            self.df_test_scaled = self.X_scaler.transform(self.df_test)

            # Split data

            self.X = self.df_scaled.copy()
            self.y = self.df_scaled.copy()
            self.X_train = self.df_train_scaled.copy()
            self.y_train = self.df_train_scaled.copy()
            self.X_test = self.df_test_scaled.copy()
            self.y_test = self.df_test_scaled.copy()

    def reshape_data(self):
        """
        Reshapes the data based on X_train, and y_train's shapes
        """
        # Get n_features
        self.X_n_features = self.X.shape[1]
        # print(self.X_train.shape)
        self.y_n_features = self.y.shape[1]
        # print(self.y_train.shape)

        # Reshape data
        self.X_reshaped = self.X.reshape((len(self.X),
                                         self.X_n_features))
        self.y_reshaped = self.y.reshape((len(self.y),
                                         self.y_n_features))



        self.X_train_reshaped = self.X_train.reshape((len(self.X_train),
                                                      self.X_n_features))
        self.y_train_reshaped = self.y_train.reshape((len(self.y_train),
                                                      self.y_n_features))

        self.X_test_reshaped = self.X_test.reshape((len(self.X_test),
                                                    self.X_n_features))
        self.y_test_reshaped = self.y_test.reshape((len(self.y_test),
                                                    self.y_n_features))

    def create_TS_generators(self, n_days):
        """
        Creates the data generators
        """
        self.n_input = n_days
        # Get data generators
        self.data_gen = sequence.TimeseriesGenerator(
                            self.X_reshaped,
                            self.y_reshaped,
                            length=self.n_input)


        # self.train_data_gen = sequence.TimeseriesGenerator(
        #                     self.X_train_reshaped,
        #                     self.y_train_reshaped,
        #                     length=self.n_input)
        # self.test_data_gen = sequence.TimeseriesGenerator(
        #                     self.X_test_reshaped,
        #                     self.y_test_reshaped,
        #                     length=self.n_input)


if __name__ == "__main__":
    df_hermes = pd.read_pickle("./data/modeling/hermes.pkl")
    df_analyst = pd.read_pickle("./data/modeling/analyst.pkl")
    X_cols = [col for col in df_hermes.columns
              if col not in df_analyst.columns]
    y_cols = [col for col in df_hermes.columns
              if col in df_analyst.columns]
    model_name = 'hermes'
    n_days = 4
    loader = NetworkLoader(df_hermes, X_cols, y_cols,
                           model_name, n_days)
    parameters = {'use_input_regularizer': 0,
              'input_neurons': 64,
              'input_dropout_rate': 0.3,
              'use_hidden_regularizer': 0,
              'hidden_dropout_rate': 0.0,
              'n_hidden_layers': 3,
              'hidden_neurons': 64,
              'patience': 50,
              'use_early_stopping': 0,
              'batch_size': 32,
              'hidden_regularizer_penalty': 0.1,
              'input_regularizer_penalty': 0.01
             }
    model = loader.build_model(**parameters)
    history = model.fit(
        loader.train_data_gen,
        validation_data=(loader.test_data_gen),
        epochs=5
    )
