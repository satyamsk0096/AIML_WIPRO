#Missing values
import pandas as pd
import numpy as np

# Creating a sample DataFrame with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, 3, 4, 5],
    'D': [np.nan, np.nan, np.nan, 4, 5]
}
df = pd.DataFrame(data)
print('Data \n', df)
# Identifying missing data
print("Count of Missing data:\n", df.isnull().sum())

#Techniques to Handle Missing Data

'''
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
}
df = pd.DataFrame(data)
'''
# Dropping rows with any missing data
df_dropped_rows = df.dropna(axis=0)
print("DataFrame after dropping rows with any missing data:\n", df_dropped_rows)

# Dropping columns with any missing data
df_dropped_cols = df.dropna(axis=1)
print("DataFrame after dropping columns with any missing data:\n", df_dropped_cols)

# Filling missing data with a specific value
df_filled = df.fillna(0)
print("DataFrame after filling missing data with 0:\n", df_filled)

'''
Imputation Methods
'''

# Imputing missing data with the mean of each column
df_mean_imputed = df.fillna(df.mean())
print("DataFrame after mean imputation:\n", df_mean_imputed)

# Imputing missing data with the median of each column
df_median_imputed = df.fillna(df.median())
print("DataFrame after median imputation:\n", df_median_imputed)

# Imputing missing data with the mode of each column
df_mode_imputed = df.fillna(df.mode().iloc[0])
print("DataFrame after mode imputation:\n", df_mode_imputed)

# Interpolating missing data
df_interpolated = df.interpolate()
print("DataFrame after interpolation:\n", df_interpolated)