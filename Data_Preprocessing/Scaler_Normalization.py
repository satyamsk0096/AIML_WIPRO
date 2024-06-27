import pandas as pd
from sklearn.preprocessing import StandardScaler

#StandardScaler
# Creating a sample DataFrame
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [10, 20, 30, 40, 50],
    'Feature3': [100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

# Applying standardization
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Standardized DataFrame:\n", df_standardized)


#MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

# Applying min-max scaling
scaler = MinMaxScaler()
df_minmax_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Min-Max Scaled DataFrame:\n", df_minmax_scaled)


#RobustScaler
from sklearn.preprocessing import RobustScaler

# Applying robust scaling
scaler = RobustScaler()
df_robust_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Robust Scaled DataFrame:\n", df_robust_scaled)
