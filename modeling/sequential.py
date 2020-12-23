import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential


class CustomSequential(Sequential):
    """A wrapper for Sequential to train on n_days, and
    implement a cross-validation of the data.

    Note: the cv does not change, so we can pick up training whenever,
    and the model will not start training on different data."""

    model = None
    nt = None

    def __init__(self, k_folds, n_days):
        """
        k_folds(int > 2)
            - number of time the data is folded for multi validation.
        n_days(int > 1)
            - number of days to predict `tomorrow` with.
        """
        self.n_days = n_days
        self.k_folds = k_folds

        # Makes a Sequential() model
        super(CustomSequential, self).__init__()

    def fit(self, nt, **kwargs):
        """
        Overrides model fit to call it k_folds times
        then averages the loss and val_loss to return back
        as the history.
        """
        # Deleting kwargs['n_days'] so it doesn't get fed
        # into the model.fit
        del kwargs['n_days']
        try:
            del kwargs['validation_data']
        except KeyError:
            pass
        # Can't set nt each iteration b/c hypertuner starts
        # feeding the data straight in, inplace of nt
        if not self.nt:
            self.nt = nt

        # TODO Run Parallel if possible
        # from joblib import Parallel
        # may not work when running on GPU would need Spark
        histories = []
        h = None

        # For resetting the model weights between cv splits
        original_weights = self.get_weights()

        # Iterate over number of k_folds
        for k in range(1, self.k_folds+1):
            train, test, val = self.nt.n_day_gens[self.n_days][k]

            # Split data and targets
            X, y = train[0]
            X_t, y_t = test[0]

            # Calling Sequential.fit() with each fold
            # print("\n\nSHAPES")
            # print(X.shape, y.shape, X_t.shape, y_t.shape)
            self.set_weights(original_weights)
            h = super(CustomSequential, self).fit(
                X, y,
                validation_data=(X_t, y_t),
                **kwargs)
            histories.append(h.history)
        # Get and return average of model histories
        df = pd.DataFrame(histories)
        h.history['loss'] = np.array(df['loss'].sum()) / len(df)
        h.history['val_loss'] = np.array(df['val_loss'].sum()) / len(df)
        return h
