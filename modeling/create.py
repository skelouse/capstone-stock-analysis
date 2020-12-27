import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from functools import partial

# This should check if it's in a jupyter environment
from IPython.display import Markdown, display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, ConfusionMatrixDisplay, \
                            confusion_matrix

# Tensorflow / keras
from tensorflow.keras.preprocessing import sequence
import kerastuner as kt
from keras import backend as K
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.regularizers import l2, l1

# Custom library for building the network
from .build import NetworkBuilder, DummyHp


class NetworkCreator():
    """
    For creating a Time Series predicting neural network
    Used for
        - tuning model parameters
        - testing tuned parameters
        - cleaning the TS data
        - delivering model reports

    Parameters
    ----------------------------------------
    df{pd.DataFrame}::
        A dataframe consisting of X_cols and y_cols
    X_cols[list, str]::
        Either uses a list to directly slice the
        data and targets, OR uses all columns that
        contain the supplied string.
    y_cols[list, str]::
        Either uses a list to directly slice the
        data and targets, OR uses all columns that
        contain the supplied string.
    n_days(int)::
        Number of days to use in each days prediction
        EX:)  if n_days was 3
            - slice targets[3:]
            - use data[0:4] to predict first target
            - data[1:5] to predict next target
            - ...
    test_split=0.3(float 0-1)::
        The decimal percentage to split the
        test data on
    val_split=0.05(float 0-1)::
        The decimal percentage to split the
        val data on

    Example Usage
    ----------------------------------------
    # Import libraries
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> import pandas as pd

    # Load dataset
    >>> flights = sns.load_dataset('flights')

    # Define n_years could also be n_days
    >>> n_years = 1

    # Map month strings to 0-12
    >>> month_map = flights['month'][:12].reset_index(drop=True).to_dict()
    >>> flights['month'] = flights['month'].map(month_map)

    # Instantiate NetworkCreator
    >>> creator = NetworkCreator(flights, 'month', 'passengers', n_years)

    # build and fit model ( With default parameters )
    >>> creator.build_and_fit_model(dummy_hp=True)

    # Fit model
    >>> history = creator.model.fit(
    >>>     creator.train_data_gen,
    >>>     validation_data=creator.val_data_gen,
    >>>     epochs=10)

    # Plot the loss and val_loss
    >>> plt.plot(history.history['loss'])
    >>> plt.plot(history.history['val_loss'])
    >>> plt.legend(['train', 'test'])
    >>> plt.show()
    """
    # TODO add splitting df like df_train/test/val
    def __init__(self, df, X_cols, y_cols, n_days,
                 test_split=0.3, val_split=0.05,
                 tuning=False, verbose=True):
        self.model = None
        self.df = df
        self.X_cols = X_cols
        self.y_cols = y_cols
        # TODO add checks that splits are between 0 and 1
        self.test_split = test_split
        self.val_split = val_split
        self.tuning = tuning
        self.verbose = verbose
        if tuning:
            self.prepare_data_gen(n_days)
        else:
            self.prepare_data(n_days)

    def prepare_data(self, n_days=1):
        """
        Runs in initialization
        calls
          - clean_cols
          - split_dataframe
          - split_and_scale_dataframes
          - reshape data
          - create_TS_generators
          - initializes input_shape
                (n_input, X_n_features)

        Parameters
        ----------------------------------------
        n_days(int)::
            Number of time periods to use in prediction

        """
        self.clean_cols()

        # Splits dataframe to X_cols + y_cols only,
        # and to train, test, val
        self.split_dataframe(self.test_split, self.val_split)

        # Get column indices
        self.column_indices = \
            {name: i for i, name in enumerate(self.df.columns)}

        # Scale data X_train, etc is not scaled
        # Scalers are created for inverse reference
        self.split_and_scale_dataframes()

        # Reshape data, creates X_train->val_reshaped
        self.reshape_data()

        # Initialize n_input
        self.n_input = n_days

        # Create Time Series Generators
        if self.val_split:
            self.train_data_gen, self.test_data_gen, self.val_data_gen = \
                self.create_TS_generators(n_days=n_days)
        elif self.tuning:
            self.data_gen = self.create_TS_generators(n_days=n_days)
        else:
            self.train_data_gen, self.test_data_gen = \
                self.create_TS_generators(n_days=n_days)

        # Define input shape for model
        self.input_shape = (self.n_input,
                            self.X_n_features)

    def prepare_data_gen(self, n_days):
        """Prepares data for the tuner, only creating
        data_gen vs before creating train/test/val data_gens."""
        self.clean_cols()

        # Splits dataframe to X_cols + y_cols only
        assert(self.split_dataframe())

        # Get column indices
        self.column_indices = \
            {name: i for i, name in enumerate(self.df.columns)}

        # Scale data X_train, etc is not scaled
        # Scalers are created for inverse reference
        assert(self.split_and_scale_dataframes())

        # Reshape data, creates X_train->val_reshaped
        assert(self.reshape_data())

        # Initialize n_input
        self.n_input = n_days

        # Create Time Series Generator
        self.data_gen = self.create_TS_generators(n_days=n_days)

        # Define input shape for model
        self.input_shape = (self.n_input,
                            self.X_n_features)

    def build_and_fit_model(self, hp=None, **parameters):
        """wrapper for build and fit model"""
        builder = NetworkBuilder(
            self,
            self.n_input,
            self.input_shape,
            output_shape=len(self.y_cols)
            )
        self.model = builder.build_and_fit_model(hp, **parameters)
        return self.model

    def clean_cols(self):
        """
        If a string is given for cols then it will take all of the columns
        that are in the dataframe that contain that string
        """
        if isinstance(self.X_cols, str):
            self.X_cols = \
                [col for col in self.df.columns if self.X_cols in col]
            if self.verbose:
                print("Got", len(self.X_cols), "X columns")

        if isinstance(self.y_cols, str):
            self.y_cols = \
                [col for col in self.df.columns if self.y_cols in col]
            if self.verbose:
                print("Got", len(self.y_cols), "y columns")

    # def _contains(self, sub, pri):
    #     """
    #     For checking if one list is inside of another
    #     https://stackoverflow.com/questions/3847386/how-to-test-if-a-list-contains-another-list  # noqa
    #     """
    #     # TODO check if .isin() would do the same thing
    #     M, N = len(pri), len(sub)
    #     i, LAST = 0, M-N+1
    #     while True:
    #         try:
    #             found = pri.index(sub[0], i, LAST)  # find first elem in sub
    #         except ValueError:
    #             return False
    #         if pri[found:found+N] == sub:
    #             return [found, found+N-1]
    #         else:
    #             i = found+1

    @classmethod
    def split_perc(cls, df, test_split=.3, val_split=.05):
        """
        Splits a dataframe into train, test, and val
        if val_split is > 0.

        Parameters
        ----------------------------------------
        test_split = 0.3 (float 0-1)
            - the percentage of data to split as test

        val_split = .05 (float 0-1)
            - the percentage of data to split as val

        Returns
        ----------------------------------------
        train
            - first portion of the data, % size of
            1 - (test_split + val_split)

        test
            - last portion of the data, % size of
            test_split

        val if val split > 0
            - middle portion of the data, % size of
            val_split

        Example Usage
        ----------------------------------------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        >>>     'apple': [1, 2, 3, 4, 5, 6],
        >>>     'orange': [1, 2, 3, 4, 5, 6]
        >>> })
        >>> train, test = NetworkCreator.split_perc(
        ...                 df, val_split=0)
        >>> print(train)
        >>> print(test)
        # train
        ...      apple  orange
        ... 0      1       1
        ... 1      2       2
        ... 2      3       3
        ... 3      4       4
        # test
        ...      apple  orange
        ... 4      5       5
        ... 5      6       6

        """
        if val_split:
            total_split = test_split + val_split
            train, test_val = train_test_split(
                df, test_size=total_split, shuffle=False)
            real_split = val_split/test_split
            val, test = train_test_split(
                test_val, train_size=real_split, shuffle=False)
            return train, val, test
        else:
            train, test = train_test_split(
                df, test_size=test_split, shuffle=False)
            return train, test

    def split_dataframe(self, test_split=.3, val_split=.05):
        """
        Splits the dataframe on defined X_cols and y_cols
        after that splitting on test_split and val_split.

        Parameters
        ----------------------------------------
        test_split{float 0-1}::
          - the percentage of the data to use for testing
        val_split{float 0-1}::
          - the percentage of the data to us for validation
        """
        try:
            if len(self.y_cols) == 1:
                if self.verbose:
                    print("target is in data")
                select_cols = self.X_cols + self.y_cols
                self.df = self.df[select_cols]

            elif self.X_cols == self.y_cols:
                if self.verbose:
                    print('target(s) equal data')
                self.df = self.df[self.X_cols]

            elif set(self.y_cols).issubset(self.X_cols):
                if self.verbose:
                    print("targets are in x")
                self.df = self.df[self.X_cols]
            # elif ((self._contains(self.y_cols, self.X_cols)[1]
            #       - self._contains(self.y_cols, self.X_cols)[0])
            #       + 1 == len(self.y_cols)):
                # if self.verbose:
                #     print("targets are in x")

            else:
                if self.verbose:
                    print('target is not in data')
                select_cols = self.X_cols + self.y_cols
                self.df = self.df[select_cols]

        except TypeError:
            if self.verbose:
                print('y is in dataframe but not x')
            select_cols = self.X_cols + self.y_cols
            self.df = self.df[select_cols]

        # Execution saver, only need data_gen when tuning
        if self.tuning:
            return 1
        # Split dataframes
        if val_split:
            train, val, test = self.split_perc(self.df, test_split, val_split)
            self.df_train = train
            self.df_test = test
            self.df_val = val
        else:
            train, test = self.split_perc(self.df, test_split, val_split=0)
            self.df_train = train
            self.df_test = test

    def split_and_scale_dataframes(self):
        """
        Scales and splits the data into X and y of each
        train, test, and val.

        Creates X and/or y scalers for inverse reference
        """

        # If there is one target
        if len(self.y_cols) == 1:

            # Define scaler
            self.X_scaler = MinMaxScaler()

            # Define y_col_idx
            self.y_col_idx = self.column_indices[self.y_cols[0]]

            # Scale data
            self.df_scaled = self.X_scaler.fit_transform(self.df)

            # Execution saver
            if self.tuning:
                self.X = self.df_scaled[:, :self.y_col_idx].copy()
                self.y = self.df_scaled[:, self.y_col_idx].copy()
                return 1

            self.df_train_scaled = self.X_scaler.transform(self.df_train)
            self.df_test_scaled = self.X_scaler.transform(self.df_test)
            if self.val_split:
                self.df_val_scaled = self.X_scaler.transform(self.df_val)

            # Split data
            self.X = self.df_scaled[:, :self.y_col_idx].copy()
            self.y = self.df_scaled[:, self.y_col_idx].copy()
            self.X_train = self.df_train_scaled[:, :self.y_col_idx].copy()
            self.y_train = self.df_train_scaled[:, self.y_col_idx].copy()
            self.X_test = self.df_test_scaled[:, :self.y_col_idx].copy()
            self.y_test = self.df_test_scaled[:, self.y_col_idx].copy()
            if self.val_split:
                self.X_val = self.df_val_scaled[:, :self.y_col_idx].copy()
                self.y_val = self.df_val_scaled[:, self.y_col_idx].copy()

        elif self.X_cols != self.y_cols:  # Accounting for another scaler
            # TODO check if this is accurate, this may just be able to tie
            #      with above
            # Split df by X and y cols
            self.X_df = self.df[self.X_cols]
            self.y_df = self.df[self.y_cols]

            # Tuning has it's own train test split
            if self.tuning:
                self.X_scaler = MinMaxScaler()
                self.y_scaler = MinMaxScaler()
                self.X = self.X_scaler.fit_transform(self.X_df)
                self.y = self.y_scaler.fit_transform(self.y_df)
                return 1

            self.X_df_train = self.df_train[self.X_cols]
            self.y_df_train = self.df_train[self.y_cols]

            self.X_df_test = self.df_test[self.X_cols]
            self.y_df_test = self.df_test[self.y_cols]

            if self.val_split:
                self.X_df_val = self.df_val[self.X_cols]
                self.y_df_val = self.df_val[self.y_cols]

            # Define scalers
            self.X_scaler = MinMaxScaler()
            self.y_scaler = MinMaxScaler()

            # Scale data
            self.X = self.X_scaler.fit_transform(self.X_df)
            self.y = self.y_scaler.fit_transform(self.y_df)

            self.X_train = self.X_scaler.transform(self.X_df_train)
            self.y_train = self.y_scaler.transform(self.y_df_train)

            self.X_test = self.X_scaler.transform(self.X_df_test)
            self.y_test = self.y_scaler.transform(self.y_df_test)

            if self.val_split:
                self.X_val = self.X_scaler.transform(self.X_df_val)
                self.y_val = self.y_scaler.transform(self.y_df_val)

        else:  # If X and y are the same i.e predicting self with self

            # Define scaler, y is 0 because it is not used
            self.X_scaler = MinMaxScaler()
            self.y_scaler = 0

            # Scale data
            self.df_scaled = self.X_scaler.fit_transform(self.df)
            self.df_train_scaled = self.X_scaler.transform(self.df_train)
            self.df_test_scaled = self.X_scaler.transform(self.df_test)
            if self.val_split:
                self.df_val_scaled = self.X_scaler.transform(self.df_val)

            # Split data
            self.X = self.df_scaled.copy()
            self.y = self.df_scaled.copy()

            self.X_train = self.df_train_scaled.copy()
            self.y_train = self.df_train_scaled.copy()

            self.X_test = self.df_test_scaled.copy()
            self.y_test = self.df_test_scaled.copy()

            if self.val_split:
                self.X_val = self.df_val_scaled.copy()
                self.y_val = self.df_val_scaled.copy()

    def reshape_data(self):
        """
        Reshapes the data based on 
          - length of X/y
          - X/y number of features
        reshape(length, n_features)
        """
        # Get n_features
        self.X_n_features = self.X.shape[1]

        if len(self.y_cols) == 1:
            self.y_n_features = 1
        else:
            self.y_n_features = self.y.shape[1]
        # print(self.y.shape)

        # Reshape data

        # Execution saver
        if self.tuning:
            self.X_reshaped = self.X.reshape((len(self.X),
                                             self.X_n_features))
            self.y_reshaped = self.y.reshape((len(self.y),
                                             self.y_n_features))
            return 1

        self.X_train_reshaped = self.X_train.reshape((len(self.X_train),
                                                      self.X_n_features))
        self.y_train_reshaped = self.y_train.reshape((len(self.y_train),
                                                      self.y_n_features))

        self.X_test_reshaped = self.X_test.reshape((len(self.X_test),
                                                    self.X_n_features))
        self.y_test_reshaped = self.y_test.reshape((len(self.y_test),
                                                    self.y_n_features))

        if self.val_split:
            self.X_val_reshaped = self.X_val.reshape((len(self.X_val),
                                                      self.X_n_features))
            self.y_val_reshaped = self.y_val.reshape((len(self.y_val),
                                                      self.y_n_features))

    def create_TS_generators(self, n_days):
        """
        Creates the data generators.

        TS = tensorflow.keras.preprocessing.sequence.TimeseriesGenerator
            - Time series generator for feeding into the model
            splits on data and targets.

        Example
        ----------------------------------------
        data =
              apple  orange  banana
        date
          1     1      2       2
          2     4      2       3
          3     3      3       6

        data = [apple, orange, banana]
        target = banana
        target is banana the following day, predicted
        with n_days before

        TS[n_days = 1]:
        Here banana on day two is predicted with all of the the data
        from day one, then day three is predicted from all of the data
        from day two
            data:
                [[1, 2, 2],
                [4, 2, 3]]
            target:
                [[3],
                [6]]
        TS[n_days = 2]:
        Here banana on day three is predicted with all of the data
        from all of the columns on day 1 and 2
            data:
                [[1, 2, 2
                  4, 2, 3]]
            target:
                [[6]]

        Parameters
        ----------------------------------------
        n_days{int}::
            Number of days to predict next with

        Returns
        ----------------------------------------
        data[TS]
        train[TS]
        test[TS]
        val[TS] if val_split > 0
        """
        if self.tuning:
            data = sequence.TimeseriesGenerator(
                self.X_reshaped,
                self.y_reshaped,
                length=self.n_input
            )
            return data

        train = sequence.TimeseriesGenerator(
            self.X_train_reshaped,
            self.y_train_reshaped,
            length=self.n_input
            )

        test = sequence.TimeseriesGenerator(
            self.X_test_reshaped,
            self.y_test_reshaped,
            length=self.n_input
            )

        if self.val_split:
            val = sequence.TimeseriesGenerator(
                self.X_val_reshaped,
                self.y_val_reshaped,
                length=self.n_input
                )
            return train, test, val

        return train, test

    def load_parameters(self, name, directory="./tuner_directory"):
        """
        Loads in the tuned parameters for building the model

        Parameters
        ----------------------------------------
        name{type}::
            The name of the tuned model
        directory(str)::
            Where to get the parameters from

        Returns
        ----------------------------------------
        parameters w/o Hyperband fit arguments

        Example Usage
        ----------------------------------------
        >>> 
        >>> 
        >>> 
        >>> 
        """
        self.build_and_fit_model = partial(
            self.build_and_fit_model, dummy_hp=True
                                             )
        tuner = kt.Hyperband(self.build_and_fit_model,
                             objective='val_loss',
                             max_epochs=1000,
                             directory=directory,
                             project_name=name)

        print("Getting top hyper-parameters for:", name)
        parameters = tuner.get_best_hyperparameters(num_trials=1)
        parameters = parameters[0].__dict__['values']
        reg_parameters = dict([(key, value)
                               for key, value in parameters.items()
                               if 'tuner' not in key])
        tuner_parameters = dict([(key, value)
                                for key, value in parameters.items()
                                if 'tuner' in key])
        print("removed tuner_parameters\n", tuner_parameters)
        return reg_parameters

    def get_r2_scores_one_y(self, _print=False):
        """
        Creates the r2 scores for data if there is a single target
        """
        def get_prediction_set(set):
            df_scaled = self.__dict__[f"df_{set}_scaled"]
            data_gen = self.__dict__[f"{set}_data_gen"]

            # Predict from data_gen
            prediction = self.model.predict(data_gen)

            # Create a copy of df_scaled to use as y_true
            temp_df = df_scaled.copy()[self.n_input:]
            y_true = temp_df[:, self.y_col_idx].copy()

            # Plugging y_pred for testing predictions
            temp_df[:, self.y_col_idx] = \
                prediction.reshape(len(prediction))

            # Inverse transforming the predictions
            self.__dict__[f"{set}_y_pred"] = self.X_scaler\
                .inverse_transform(temp_df)[:, self.y_col_idx]

            # Plug y_true back into dataframe, and inverse transform
            temp_df = df_scaled.copy()[self.n_input:]
            temp_df[:, self.y_col_idx] = y_true
            self.__dict__[f"{set}_y_true"] = self.X_scaler\
                .inverse_transform(temp_df)[:, self.y_col_idx]

            # set the r2 score from y_true and y_pred
            # using sklearn.metrics.r2_score
            r2 = r2_score(self.__dict__[f"{set}_y_true"],
                          self.__dict__[f"{set}_y_pred"])
            self.__dict__[f"{set}_r2"] = r2
            if _print:
                print(f"{set}_r2:{r2: .2f}")

        _sets = ['train', 'test']
        if self.val_split:
            _sets.append('val')

        for _set in _sets:
            get_prediction_set(_set)

    def get_r2_scores_multi_y(self, same=False):
        """
        Creates the r2 scores for multiple targets whether inside
        X_cols or not

        Parameters
        ----------------------------------------
        same(bool)::
          - whether or not X_cols and y_cols are the same
        """
        def get_prediction_set(set):
            data_gen = self.__dict__[f"{set}_data_gen"]
            if same:
                df = self.__dict__[f"df_{set}"]
            else:
                df = self.__dict__[f"y_df_{set}"]

            # Getting data_gen prediction
            prediction = self.model.predict(data_gen)

            # Inverse scaling prediction
            prediction_iv = self.y_scaler.inverse_transform(prediction)

            # Getting the y index, slicing out the non predicted values
            prediction_idx = df[self.n_input:].index
            self.__dict__[f"df_predict_{set}"] = \
                pd.DataFrame(prediction_iv,
                             columns=self.y_cols,
                             index=prediction_idx)

            y_pred = self.__dict__[f"df_predict_{set}"]
            # Getting y_true by locing from df the index of df_predict
            y_true = df.loc[y_pred.index]

            # Setting y_true and pred
            self.__dict__[f"{set}_y_true"] = y_true
            self.__dict__[f"{set}_y_pred"] = y_pred

            # Calculating r2_score with sklearn.metrics.r2_score
            r2 = r2_score(y_true, self.__dict__[f"df_predict_{set}"])

            # Setting set_r2
            self.__dict__[f"{set}_r2"] = r2

        _sets = ['train', 'test']
        if self.val_split:
            _sets.append('val')

        for _set in _sets:
            get_prediction_set(_set)

    def predict_r2_scores(self, _print=False):
        """
        desc

        Parameters
        ----------------------------------------
        name{type}::
            desc

        Returns
        ----------------------------------------
        self

        Example Usage
        ----------------------------------------
        >>> 
        >>> 
        >>> 
        >>> 
        """

        # If predicting one column for y
        if (len(self.y_cols) == 1):
            self.get_r2_scores_one_y(_print)

        # If predicting multiple columns for y
        elif self.y_scaler:
            self.get_r2_scores_multi_y()

        # if predicting self with self
        else:
            self.get_r2_scores_multi_y(same=True)

    def display_r2_scores(self, plot=False):
        if not plot:
            print("-"*20, "\nr2 scores\n" + "-"*20)
            print(f"Training:{self.train_r2:.2f}")
            print(f"Testing:{self.test_r2:.2f}")
            if self.val_split:
                print(f"Validation:{self.val_r2:.2f}")
        else:
            return self.plot_predictions()

    def plot_predictions(self, sets=None, marker="."):
        """
        Currently only works with one y

        Parameters
        ----------------------------------------
        sets[list]:: Optional
            which sets to plot
            ['train', 'test']
            OR
            ['train', 'test', 'val']
        marker(str) = "."::
            - what marker to use on the plots

        Returns
        ----------------------------------------
        fig

        Example Usage
        ----------------------------------------
        >>> 
        >>> 
        >>> 
        >>> 
        """
        # TODO fill in example usage
        # TODO find a way to plot with multi_y columns
        if self.val_split and not sets:
            sets = ['train', 'test', 'val']
        elif not sets:
            sets = ['train', 'test']
        height = ((len(sets)) * 3) + 3
        width = ((len(sets)) * 5)
        fig, axes = plt.subplots(figsize=(width, height),
                                 nrows=len(sets)+1, ncols=2,
                                 constrained_layout=True,
                                 squeeze=False, gridspec_kw=dict(hspace=.5))

        gs = axes[0][0].get_gridspec()
        for ax in axes[len(sets)]:
            ax.remove()

        _preds = []
        _Xs = []
        for num, _set in enumerate(sets):
            ax1, ax2 = axes[num]
            markersize = (num+1.5)*3

            _idx = self.__dict__[f"df_{_set}"].index
            _real = self.__dict__[f"df_{_set}"][self.y_cols]
            _true = self.__dict__[f"{_set}_y_true"]
            _pred = self.__dict__[f"{_set}_y_pred"]
            _preds.append(_pred)
            _r2 = self.__dict__[f"{_set}_r2"]
            ax1.plot(_idx, _real, marker=marker,
                     color='C0', label='Real Price',
                     markersize=markersize)
            _X = self.__dict__[f"df_{_set}"].index[self.n_input:]
            _Xs.append(_X)
            ax1.plot(_X, _true, marker=marker, color='C1', label='y_true',
                     markersize=markersize)
            ax1.plot(_X, _pred, marker=marker, color='C2', label='y_pred',
                     linewidth=3.5, markersize=markersize, alpha=.7)

            ax2.plot(_X, _pred, marker=marker, color='C2', linewidth=3.5,
                     markersize=markersize, alpha=.7)

            # Plot styling
            #    Left side
            ax1.set_title(f"""{_set.title()} data with an r2 of: {
                            _r2: .2f}""", fontsize=20)
            ax1.legend()
            length = len(_X)

            #    Right side
            ax2.set_title("Predicted", fontsize=20)

            #    Both
            for ax in axes[num]:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(
                                            int(length/4)))
                plt.setp(ax.get_xticklabels(), ha="right", rotation=20)
                ax.set_ylabel("Price", fontsize=15)

        ax = fig.add_subplot(gs[len(sets), :])

        X = self.df.index
        ax.plot(X, self.df[self.y_cols],
                label='y_true', linewidth=3)

        for name, pred, idx in zip(sets, _preds, _Xs):
            ax.plot(idx, pred, label=f"{name}-pred", alpha=.6, linewidth=4)
        ax.legend()
        ax.set_title("Actual VS predicted", fontsize=20)
        ax.set_ylabel("Price", fontsize=15)
        return fig

    def get_shap_values(self):
        """
        Creates the shap values with val_data_gen
        """
        # TODO add _set parameter for taking val, train, or test
        first = int(self.val_data_gen.data.shape[0]/self.n_input)
        last = self.val_data_gen.data.shape[1]

        explainer = shap.GradientExplainer(self.model,
                                           self.val_data_gen.data)
        self.shap_val = explainer.shap_values(
            self.val_data_gen.data.reshape((first, self.n_input, last)), 1)

    def plot_shap_summary(self):
        """
        Plots the total weight of all columns on the prediciton

        Returns
        ----------------------------------------
        fig

        Example Usage
        ----------------------------------------
        >>> 
        >>> 
        >>> 
        >>> 
        """
        # TODO add example usage
        try:
            shap_val_total = self.shap_val[0].sum(axis=1)
        except AttributeError:
            self.get_shap_values()
            shap_val_total = self.shap_val[0].sum(axis=1)
        shap.summary_plot(
            shap_val_total,
            feature_names=list(self.X_cols),
            plot_type='bar',
            show=False)
        fig = plt.gcf()
        plt.tight_layout()
        plt.show()
        return fig

    def plot_shap_bar(self):
        """
        Plots the negative/postive weights of columns on the data

        Returns
        ----------------------------------------
        fig

        Example Usage
        ----------------------------------------
        >>> 
        >>> 
        >>> 
        >>> 
        """
        # TODO add example usage
        try:
            shap_val_x_day = np.array(self.shap_val) \
                .sum(axis=1).sum(axis=1)[0]
        except AttributeError:
            self.get_shap_values()
            shap_val_x_day = np.array(self.shap_val) \
                .sum(axis=1).sum(axis=1)[0]
        shap.bar_plot(
            shap_val_x_day,
            feature_names=list(self.X_cols),
            show=False)
        fig = plt.gcf()
        plt.tight_layout()
        plt.show()
        return fig

    def classify_set(self, _set):
        """
        Used in classify to build a confusion matrix with
        y_true and y_pred after normalizing the values to 1/0, up/down
        """
        true = self.__dict__[f'{_set}_y_true']
        pred = self.__dict__[f'{_set}_y_pred']
        df = pd.DataFrame({
            'true': true,
            'pred': pred
        })
        df = df.diff().dropna()
        func = (lambda x: 1 if x > 0 else 0)
        df['true'] = df['true'].apply(func)
        df['pred'] = df['pred'].apply(func)
        r2 = r2_score(df['true'], df['pred'])
        return confusion_matrix(df['true'], df['pred'], normalize='all'), r2

    def build_classification(self, cm, ax, classes=['Down', 'Up']):
        """
        Used in classify to build the classification plot
        """
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=classes)
        return disp.plot(include_values=True, cmap='Blues', ax=ax)

    def classify(self):
        """
        Builds a classification table for train, test, and val

        Returns
        ----------------------------------------
        fig

        Example Usage
        ----------------------------------------
        >>> 
        >>> 
        >>> 
        >>> 
        """
        # TODO add example usage
        title_map = self.get_title_map()
        n = len(title_map.keys())
        fig, axes = plt.subplots(
            figsize=(15, 4), ncols=n, tight_layout=True)
        for (_set, title), ax in zip(title_map.items(), axes):
            cm, r2 = self.classify_set(_set)
            self.build_classification(cm, ax=ax)
            ax.set_title(f"{title}\nr2: {r2: .2f}",
                         fontsize=15)
        return fig

    def get_title_map(self):
        """
        Returns a mapping of
        train -> Training
        test -> Testing
        val -> Validation
        For use in plotting titles, and reports
        """
        if self.val_split:
            title_map = {
                'train': "Training",
                'test': "Testing",
                'val': "Validation"
            }
        else:
            title_map = {
                'train': "Training",
                'test': "Testing"
            }
        return title_map

    def display_and_save_report(
            self,
            name,
            save=True,
            transparent_plots=False,
            save_model=True,
            show_headers=True,
            filepath="./reports",
            display_notebook=True):
        """
        Makes an HTML report of all of the different classification
        and prediction plots

        Parameters
        ----------------------------------------
        name(str)::
            - The name of the report
        save=True(bool)::
            - Whether to save the HTML report or not
        transparent_plots=False(bool)::
            - Whether to save as jpg or png with transparent BG
        save_model=True(bool)::
            - Whether to save the model itself or not
        show_headers=True(bool)::
            - Whether to display the html headers on the plots or not
        filepath="./reports"(str)::
            - The filepath of where the report is saved
        display_notebook=True(bool)::
            - Whether or not to use notebook display or python

        Example Usage
        ----------------------------------------
        >>> 
        >>> 
        >>> 
        >>> 
        """
        # TODO add history
        # TODO implement display_notebook=False
        headers = [
            "Predictions",
            "Classification",
            "Summary importances",
            "Bar importances"
        ]
        headers.reverse()

        def header():
            "Used for making the html headers"
            _name = headers.pop()
            _str = f"""
<h3 style="
    color:White;
    background-image: linear-gradient(180deg, SlateBlue, rgb(1, 1, 1));
    text-align: center;
    margin-top: 1em
    position: center;
    left: 9.5em;
    border: 3px solid LightGray;
    "
>{_name}</h3>
            """
            display(Markdown(_str))

        # Predictions
        header()
        self.predict_r2_scores()
        pred_fig = self.plot_predictions()
        plt.show()

        # Classifications
        header()
        cm_fig = self.classify()
        plt.show()

        # Shap summary
        header()
        shap_summary_fig = self.plot_shap_summary()

        # Shap bar
        header()
        shap_bar_fig = self.plot_shap_bar()
        if save:
            filepath = filepath + f"/{name}/"
            img_filepath = filepath + "img/"

            # Check if directories exist
            if not os.path.isdir(filepath):
                os.mkdir(filepath)
            if not os.path.isdir(img_filepath):
                os.mkdir(img_filepath)

            # Define plot extension
            if transparent_plots:
                ext = '.png'
            else:
                ext = '.jpg'

            # Create plot_names for naming the given plots while saving
            plot_names = {
                pred_fig: "prediction_scoring",
                cm_fig: "classification_scoring",
                shap_summary_fig: "summary_importances",
                shap_bar_fig: "bar_importances"
            }

            # Save figures
            for fig, _name in plot_names.items():
                fig.savefig(img_filepath + _name + ext,
                            transparent=transparent_plots)

        if save_model:
            # TODO add pathfinding
            self.model.save(f"./data/models/{name}.h5")
        return 1

    def old_build_and_fit_model(
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
        """
        desc

        Parameters
        ----------------------------------------
        name{type}::
            desc

        Returns
        ----------------------------------------
        self

        Example Usage
        ----------------------------------------
        >>> 
        >>> 
        >>> 
        >>> 
        """

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
        input_neurons = self.n_input*hp.Choice('input_neurons', input_neurons)
        self.model.add(LSTM(input_neurons,
                            input_shape=self.input_shape,
                            kernel_regularizer=_reg))

        #           Dropout layer
        input_dropout_rate = hp.Choice('input_dropout_rate',
                                       input_dropout_rate)
        if input_dropout_rate != 0:
            self.model.add(Dropout(input_dropout_rate))

        self.model.add(GaussianNoise(1))

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
        use_early_stopping = hp.Choice('use_early_stopping',
                                       use_early_stopping)
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


if __name__ == "__main__":
    # Import libraries
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Load dataset
    flights = sns.load_dataset('flights')

    # Define n_years could also be n_days
    n_years = 1

    # Map month strings to 0-12
    month_map = flights['month'][:12].reset_index(drop=True).to_dict()
    flights['month'] = flights['month'].map(month_map)

    # Instantiate NetworkCreator
    creator = NetworkCreator(flights, 'month', 'passengers', n_years)

    # build and fit model ( With default parameters )
    creator.build_and_fit_model(dummy_hp=True)

    # Fit model
    history = creator.model.fit(
        creator.train_data_gen,
        validation_data=creator.val_data_gen,
        epochs=10
    )

    # Plot the loss and val_loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'test'])
    plt.show()
