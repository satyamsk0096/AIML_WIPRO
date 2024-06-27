import pandas as pd

#Creating New Features
# Sample data
data = {
    'Order Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'Delivery Date': ['2023-01-05', '2023-01-06', '2023-01-07'],
    'Product Price': [100, 150, 200]
}

df = pd.DataFrame(data)
print('ODF \n', df)
# Converting columns to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Delivery Date'] = pd.to_datetime(df['Delivery Date'])

# Creating new feature 'Delivery Time'
df['Delivery Time'] = (df['Delivery Date'] - df['Order Date']).dt.days

print('AFE \n',df)

#Polynomial Features -- Housing Price Prediction

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Sample data
data = {
    'Size (sq ft)': [1500, 2050, 2400],
    'Number of Bedrooms': [3, 4, 5]
}

df = pd.DataFrame(data)

# Creating polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df)

# Creating a DataFrame with the polynomial features
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(df.columns))

print(poly_df)

#Interaction Features -- Customer Demographics and Purchase Behavior

import pandas as pd

# Sample data
data = {
    'Age': [25, 35, 45],
    'Income': [50000, 75000, 100000]
}

df = pd.DataFrame(data)

# Creating interaction feature 'Age * Income'
df['Age * Income'] = df['Age'] * df['Income']

print(df)
