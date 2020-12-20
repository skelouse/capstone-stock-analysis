import os
import copy
import shutil
import kerastuner as kt
from functools import partial
from .create import NetworkCreator
from sklearn.model_selection import TimeSeriesSplit


class Logger():
    """A logger used for clearing the checkpoints every
    five iterations as it will start to take up TB of data
    as the training grows."""
    x = 0

    @classmethod
    def clear_checkpoints(cls):
        """Clears the checkpoints in
        './tuner_directory{name}/trial_x/chechpoints only keeping
        the information about previously tested parameters."""
        print("Clearing checkpoints")
        project_files = os.listdir(cls.filepath)
        for file in project_files:
            try:
                shutil.rmtree(cls.filepath+file+'/checkpoints')
            except FileNotFoundError:
                pass

    @classmethod
    def report_trial_state(cls, *args):
        """Function that is called by HyperBand after each trial
        used to call `clear_checkpoints` every five iterations"""
        cls.x += 1
        print("trial", cls.x)
        if cls.x >= 5:
            cls.clear_checkpoints()
            cls.x = 0

    @classmethod
    def register_directory(cls, name):
        cls.filepath = f'./tuner_directory/{name}/'

    def register_tuner(*args):
        "Runs when starting tuner"
        pass

    def register_trial(*args):
        "Runs before each trial"
        pass


class NetworkTuner(NetworkCreator):
    """
    - able to tune all things including n_days
    - must keep a batch of val_data separated for every
    sequence, and after everything is tuned it reports
    back on the val data

    - for each iteration supplies a random split for
      cross validation override model fit

    - (1/0) for each column to tune which columns to use

    Creates
    -----------------------------------------------------
    self.TS_gens[list]
       [(train, test, val), (train, test, val), ...] n k_folds
    """
    def __init__(self, df, X_cols, y_cols, k_folds=5, max_n_days=3):
        self.df = df
        self.X_cols = X_cols
        self.y_cols = y_cols
        self.k_folds = k_folds
        self.tscv = TimeSeriesSplit(n_splits=k_folds)
        self.create_k_folds(max_n_days, k_folds)
        super(NetworkTuner, self).__init__(
            df, X_cols, y_cols, 1, tuning=True, verbose=0
        )

    def split_TS(self, TS_gen):
        """Gets k self.k_folds splits of self.df

        Parameters
        ----------------------------------------
        TS_gen{tensorflow.keras.preprocessing..sequenceTimeseriesGenerator}::
            A time series data generator of length n_days

        Returns
        ----------------------------------------
        tuple of lists
        (
        ([train_data_gen]),
            - the training data for the model.
        ([test_data_gen]),
            - the val data for the model.
        ([val_data_gen])
            - the data for validating the model after running the tuner.
        )
        """
        X_train, y_train = None, None
        X_test, y_test = None, None
        X_val, y_val = None, None

        # Data to be split
        X, y = TS_gen[0]

        # (Train - val) / test split      Splitting on X
        for train_index, test_index in self.tscv.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        # Train / val split              Splitting on X_train
        for train_index, val_index in self.tscv.split(X_train):

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

        train_data_gen = (X_train, y_train)
        test_data_gen = (X_test, y_test)
        val_data_gen = (X_val, y_val)
        return (
            ([train_data_gen]),
            ([test_data_gen]),
            ([val_data_gen])
            )

    def create_k_folds(self, max_n_days, k_folds, random_state=None):
        """Creates all the random fold data for """
        self.n_day_gens = {}
        self.TS_gens = {}
        # Define tuning Network Creator
        for n_days in range(1, max_n_days+1):
            creator = NetworkCreator(
                self.df, self.X_cols, self.y_cols, n_days,
                tuning=True, verbose=0
            )
            for k in range(1, k_folds+1):
                self.TS_gens[k] = self.split_TS(creator.data_gen)
            self.n_day_gens[n_days] = copy.deepcopy(self.TS_gens)

    def tune(self, name, max_epochs=10, **parameters):
        """Running the tuner with kerastuner.Hyperband"""
        self.build_and_fit_model = partial(
            self.build_and_fit_model, **parameters
        )
        Logger.register_directory(name)
        tuner = kt.Hyperband(self.build_and_fit_model,
                             objective='val_loss',
                             max_epochs=max_epochs,
                             factor=3,
                             directory='./tuner_directory',
                             project_name=name,
                             logger=Logger)

        tuner.search(self)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""The hyperparameter search is complete.
        The optimal number of units in the first densely-connected layer
        {best_hps.__dict__['values']}
        """)
