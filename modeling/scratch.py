import pandas as pd
import copy
# from .create import NetworkCreator
# from .build import NetworkBuilder
# from .tuner import NetworkTuner
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.preprocessing import sequence

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
X_cols = df.columns
y_cols = 'banana'
#creator = NetworkTuner(df, X_cols, y_cols)
tscv = TimeSeriesSplit(n_splits=5)
data = sequence.TimeseriesGenerator(
    df[X_cols],
    df[y_cols],
    length=1
)



from sklearn.model_selection import GridSearchCV