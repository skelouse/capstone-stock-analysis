import copy

from numpy.lib.utils import deprecate
from .create import NetworkCreator
from sklearn.model_selection import TimeSeriesSplit


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
        """Gets n k_folds splits of self.df"""
        X_train, y_train = None, None
        X_test, y_test = None, None
        X_val, y_val = None, None

        X, y = TS_gen[0]
        # (Train - val) / test split
        for train_index, test_index in self.tscv.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        test_data_gen = (X_test, y_test)

        # Train / val split
        for train_index, val_index in self.tscv.split(X):

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

        train_data_gen = (X_train, y_train)
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
