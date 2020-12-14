import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras import initializers

model = Sequential()
model.compile()
model.fit()


def test_func(unchanged_list=[]):
    print("list_id:", id(unchanged_list))
    unchanged_list.append(1)
    return unchanged_list

print(test_func())
print(test_func())
print(test_func())