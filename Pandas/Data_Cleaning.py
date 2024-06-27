
'''
Data Cleaning
Handling Missing Data (dropna, fillna)
'''

import pandas as pd

# Sample DataFrame with missing values
data = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': [1, 2, 3, None]
})

# Dropping rows with any missing values
cleaned_data_drop = data.dropna()
print('cleaned \n',cleaned_data_drop)

# Filling missing values with 0
cleaned_data_fill = data.fillna(0)
print(cleaned_data_fill)

# Sample DataFrame with duplicates
data = pd.DataFrame({
    'A': [1, 2, 2, 4],
    'B': [5, 6, 6, 8]
})

# Removing duplicate rows
cleaned_data = data.drop_duplicates()
print(cleaned_data)

# Sample DataFrame
data = pd.DataFrame({
    'A': ['1', '2', '3', '4']
})
print('before \n',data)
print(data.dtypes)
# Converting data type of column 'A' to integer
data['A'] = data['A'].astype(int)
print('after \n',data)
print(data.dtypes)

# Sample DataFrame
data = pd.DataFrame({
    'A': ['Hello', 'World', 'Pandas', 'Python']
})

# Converting column 'A' to lowercase
data['A'] = data['A'].str.lower()
print(data)


'''
Data Transformation
Applying Functions to Data (apply, map, applymap)
'''

# Sample DataFrame
data = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
})

# Applying a function to column 'A'
data['A'] = data['A'].apply(lambda x: x * 2)
print(data)

# Mapping values in column 'B'
data['B'] = data['B'].map({5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight'})
print(data)

# Applying a function to the entire DataFrame using map
# data = data.applymap(str)
# print('str tra',data)
# print(data.dtypes)

# Sample DataFrame
data = pd.DataFrame({
    'A': [3, 1, 4, 2],
    'B': [5, 6, 7, 8]
})

# Sorting data by column 'A'
sorted_data = data.sort_values(by='A')
print(sorted_data)

data = pd.DataFrame({
    'A': [3, 1, 2, 2],
    'B': [5, 6, 7, 8]
})

# Ranking data in column 'A'
data['rank'] = data['A'].rank()
print(data)

# Sample DataFrame
data = pd.DataFrame({
    'A': [5, 15, 25, 35, 45, 4, 44,23,38,24]
})

# Binning data into discrete intervals
bins = [0, 10, 20, 30, 40, 50]
labels = ['0-10', '10-20', '20-30', '30-40', '40-50']
data['binned'] = pd.cut(data['A'], bins=bins, labels=labels)
print(data)