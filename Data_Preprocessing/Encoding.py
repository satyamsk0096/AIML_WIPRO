import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# One-Hot Encoding Example
data_one_hot = {
    'City': ['New York', 'Paris', 'Berlin', 'New York', 'Berlin']
}
df_one_hot = pd.DataFrame(data_one_hot)
print(df_one_hot)

# Using pandas get_dummies for One-Hot Encoding
df_one_hot_pd = pd.get_dummies(df_one_hot, columns=['City'])
print("One-Hot Encoded DataFrame using pandas:\n", df_one_hot_pd)

# Using sklearn's OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df_one_hot[['City']])
print('before \n',one_hot_encoded)
df_one_hot_sklearn = pd.DataFrame(one_hot_encoded,
                                  columns=encoder.get_feature_names_out(['City']))
print("One-Hot Encoded DataFrame using sklearn:\n", df_one_hot_sklearn)

# Label Encoding Example
data_label = {
    'City': ['New York', 'Paris', 'Berlin', 'New York', 'Berlin']
}
df_label = pd.DataFrame(data_label)

# Applying LabelEncoder
label_encoder = LabelEncoder()
df_label['City_Label'] = label_encoder.fit_transform(df_label['City'])
print("Label Encoded DataFrame:\n", df_label)

# Ordinal Encoding Example
data_ordinal = {
    'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small']
}
df_ordinal = pd.DataFrame(data_ordinal)

# Applying OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])
df_ordinal['Size_Ordinal'] = ordinal_encoder.fit_transform(df_ordinal[['Size']])
print("Ordinal Encoded DataFrame:\n", df_ordinal)