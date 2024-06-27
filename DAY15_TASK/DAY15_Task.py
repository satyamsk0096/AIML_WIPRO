'''Assignment 1: Implement a Bagging Regressor Using PyTorch
Dataset: Diabetes Dataset

Objective: Implement a Bagging regressor to predict diabetes progression using the Diabetes dataset. Evaluate the model using various regression metrics (MAE, MSE, RMSE, R-squared).

Steps:

Load the Diabetes dataset:
Use sklearn.datasets.load_diabetes to load the dataset.

Preprocess the data:
Normalize the features using StandardScaler.

Define the model:
Create a simple linear regression model using PyTorch.

Implement Bagging:
Use sklearn.ensemble.BaggingRegressor to implement the Bagging ensemble method.
Train multiple models on different subsets of the training data.

Evaluate the model:
Calculate and interpret regression metrics (MAE, MSE, RMSE, R-squared) on the test set.
'''
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Load the Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Implement Bagging using sklearn.ensemble.BaggingRegressor
# Define the Bagging Regressor
bagging_model = BaggingRegressor(n_estimators=10,  # Number of base estimators (models)
                                 random_state=42)

# Train the Bagging Regressor
bagging_model.fit(X_train, y_train)

# Step 3: Evaluate the model
# Predict on the test set
y_pred = bagging_model.predict(X_test)

# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # or mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R2): {r2:.2f}')



print('****************************************Assignment 2******************************************************')

'''
Assignment 2: Implement a Gradient Boosting Regressor Using PyTorch
Dataset: Any Housing Dataset

Objective: Implement a Gradient Boosting regressor to predict housing prices using the  Housing dataset. Evaluate the model using various regression metrics (MAE, MSE, RMSE, R-squared).

Steps:
Load the Housing dataset:

Preprocess the data:
Normalize the features using StandardScaler.

Define the model:
Create a simple linear regression model using PyTorch.

Implement Gradient Boosting:
Use sklearn.ensemble.GradientBoostingRegressor to implement the Gradient Boosting ensemble method.
Train the model on the training data.

Evaluate the model:
Calculate and interpret regression metrics (MAE, MSE, RMSE, R-squared) on the test set.'''

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Step 1: Load and preprocess the dataset

# Load the California housing dataset
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Step 2: Define a simple linear regression model using PyTorch

class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Step 3: Implement Gradient Boosting using sklearn.ensemble.GradientBoostingRegressor

# Initialize Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
gb_regressor.fit(X_train_scaled, y_train)

# Step 4: Evaluate the model

# Predict on the test set
y_pred = gb_regressor.predict(X_test_scaled)

# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R-squared: {r2:.2f}')
