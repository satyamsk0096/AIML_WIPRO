
'''
Merging and Joining DataFrames
'''

import pandas as pd

# Creating sample DataFrames
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3'],
    'C': ['C0', 'C1', 'C2', 'C3'],
    'D': ['D0', 'D1', 'D2', 'D3']
}, index=[0, 1, 2, 3])

df2 = pd.DataFrame({
    'A': ['A4', 'A5', 'A6', 'A7'],
    'B': ['B4', 'B5', 'B6', 'B7'],
    'C': ['C4', 'C5', 'C6', 'C7'],
    'D': ['D4', 'D5', 'D6', 'D7']
}, index=[4, 5, 6, 7])

# Concatenating DataFrames
result = pd.concat([df1, df2])
print("Concatenated DataFrame:\n", result)

#Merging on keys (merge)

left = pd.DataFrame({
    'key': ['K0', 'K1', 'K2', 'K3'],
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3']
})

right = pd.DataFrame({
    'key': ['K0', 'K1', 'K4', 'K5'],
    'C': ['C0', 'C1', 'C2', 'C3'],
    'D': ['D0', 'D1', 'D2', 'D3']
})

# Merging DataFrames
result = pd.merge(left, right, on='key')
print("Merged DataFrame:\n", result)

#Joining DataFrames (join)
# Creating sample DataFrames with different indexes
dfj1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
}, index=['K0', 'K1', 'K2'])

dfj2 = pd.DataFrame({
    'C': ['C0', 'C2', 'C3'],
    'D': ['D0', 'D2', 'D3']
}, index=['K0', 'K2', 'K3'])

# Joining DataFrames
result = dfj1.join(dfj2, how='outer')
print("Joined DataFrame:\n", result)


#Grouping and Aggregation
#Grouping data (groupby)

import pandas as pd
import numpy as np

# Creating a sample DataFrame
df = pd.DataFrame({
    'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
    'B': ['one', 'one', 'two', 'two', 'one', 'one', 'two', 'two'],
    'C': [1, 2, 3, 4, 5, 6, 7, 8],
    'D': np.random.randn(8)
})

# Grouping by column 'A'
grouped = df.groupby('A')
# Aggregating numeric data with mean
mean_result = grouped[['C', 'D']].mean()
print("Grouped DataFrame mean:\n", mean_result)