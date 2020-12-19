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


nt = NetworkTuner(df, X_cols, y_cols, k_folds=2)
print("GET DATA")
train, test, val = nt.get_data_n_days(1)
print("TRAIN")
print(train)
print(len(train))
X, y = train[0]
print("X_TRAIN")
print(X)
print("y_train")
print(y)

X_test, y_test = test[0]

print("GET DATA length")
# print(len(nt.get_data_n_days(1)))

creator = NetworkCreator(df, X_cols, y_cols, n_days=1, val_split=0)
model = creator.build_and_fit_model(dummy_hp=True, epochs=1)
model.fit(X, y, validation_data=(X_test, y_test))
