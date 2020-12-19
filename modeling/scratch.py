import pandas as pd
from create import NetworkCreator


df = pd.DataFrame({
    'apple': [1, 2, 3, 4, 5, 6],
    'orange': [1, 2, 3, 4, 5, 6],
    'banana': [1, 2, 3, 4, 5, 6],
    'pear': [1, 2, 3, 4, 5, 6]
})
X_cols = ['apple', 'orange']
y_cols = ['apple', 'orange', 'banana']
creator = NetworkCreator()
df = NetworkCreator().get_df_cols(df, X_cols, y_cols)
print(df)