import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
# from joblib import Parallel


class CustomSequential(Sequential):
    model = None
    nt = None
    # TODO fit with each cross_val
    # Return average loss

    def __init__(self, k_folds):
        """
        - Takes n k_fold
        - Sets self.data to a list of tuples
            [(train, val, test), (train, val, test)]
        - fits model on train and test
        - checks val after tuning is done for real results
        """
        self.k_folds = k_folds
        # for k in range(self.k_fold):
        #     pass
        super(CustomSequential, self).__init__()

    def fit(self, nt, **kwargs):
        """Overrides model fit to call it k_folds times
        then averages the loss and val_loss to return back
        as the history"""
        print("ARGS")
        n_days = kwargs['n_days']
        del kwargs['n_days']
        try:
            del kwargs['validation_data']
        except KeyError:
            pass
        if not self.nt:
            self.nt = nt

        # TODO Run Parallel if possible
        # from . import Parallel
        histories = []
        h = None
        for k in range(1, self.k_folds+1):
            train, test, val = self.nt.n_day_gens[n_days][k]
            X, y = train[0]
            X_t, y_t = test[0]
            h = super(CustomSequential, self).fit(
                X, y,
                validation_data=(X_t, y_t),
                **kwargs)
            histories.append(h.history)

        df = pd.DataFrame(histories)
        h.history['loss'] = np.array(df['loss'].sum()) / len(df)
        h.history['val_loss'] = np.array(df['val_loss'].sum()) / len(df)
        return h  # average of histories
