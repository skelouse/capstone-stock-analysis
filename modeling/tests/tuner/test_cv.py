import os
import pandas as pd
import copy
from modeling.tuner import NetworkTuner
from modeling.create import NetworkCreator

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

X_cols = list(df.columns)
y_cols = 'banana'


for i in range(2, 6):
    nt = NetworkTuner(df, X_cols, y_cols, k_folds=i)
    filename = f"./modeling/tests/tuner/val_folds/{i}_folds.txt"
    with open(filename, 'r') as f:
        assert(f.read() == str(nt.n_day_gens))

# TEST CREATION
# for i in range(2, 6):
#     nt = NetworkTuner(df, X_cols, y_cols, k_folds=i)
#     filename = f"./modeling/tests/tuner/val_folds/{i}_folds.txt"
#     with open(filename, 'w') as f:
#         f.write(str(nt.n_day_gens))

