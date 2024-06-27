'''DAY 14


Exercise 1: Implement Grid Search on a Regression Model Using PyTorch
Dataset: California Housing Dataset

Objective: Implement grid search to tune hyperparameters of a regression model using the California Housing dataset. Evaluate the model using various regression metrics (MAE, MSE, RMSE, R-squared).

Steps:
Load the California Housing dataset:
Use sklearn.datasets.fetch_california_housing to load the dataset.

Preprocess the data:
Normalize the features using StandardScaler.

Define the model and hyperparameters:
Create a simple linear regression model using PyTorch.
Define a grid of hyperparameters to search (e.g., learning rate, number of epochs).

Implement grid search:
Use sklearn.model_selection.GridSearchCV to perform grid search.
Evaluate the model using cross-validation metrics.

Report the best hyperparameters and model performance:
Calculate and interpret regression metrics (MAE, MSE, RMSE, R-squared) for the best model.
'''
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin


# Step 1: Load the California Housing dataset and preprocess the data

# Load California Housing dataset
california_housing = fetch_california_housing()
X, y = california_housing.data, california_housing.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Step 2: Define the PyTorch model

class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

# Step 3: Create a wrapper class for PyTorch model to make it compatible with scikit-learn

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model_class, criterion, optimizer_class, lr, epochs):
        self.model_class = model_class
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.epochs = epochs
        self.model = None  # To be initialized in fit method

    def fit(self, X, y):
        self.model = self.model_class(input_dim=X.shape[1])
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs.squeeze(), y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        return outputs.numpy()

    def get_params(self, deep=True):
        return {
            'model_class': self.model_class,
            'criterion': self.criterion,
            'optimizer_class': self.optimizer_class,
            'lr': self.lr,
            'epochs': self.epochs,
        }

# Step 4: Implement grid search with the wrapper class

param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'epochs': [100, 200, 300]
}

model = PyTorchRegressor(model_class=LinearRegression,
                         criterion=nn.MSELoss(),
                         optimizer_class=torch.optim.SGD,
                         lr=0.01,
                         epochs=100)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_tensor, y_train_tensor)

# Step 5: Evaluate the best model

best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

with torch.no_grad():
    y_pred = best_model.predict(X_test_tensor)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R-squared: {r2:.2f}")




print("=================================================EX 2===============================================")

'''
Exercise 2: Implement Random Search on a Regression Model Using PyTorch
Dataset: Boston Housing Dataset

Objective: Implement random search to tune hyperparameters of a regression model using the Boston Housing dataset. Evaluate the model using various regression metrics (MAE, MSE, RMSE, R-squared).

Steps:
Load the Boston Housing dataset:
Use sklearn.datasets.load_boston to load the dataset.

Preprocess the data:
Normalize the features using StandardScaler.

Define the model and hyperparameters:
Create a simple linear regression model using PyTorch.
Define a range of hyperparameters to search (e.g., learning rate, number of epochs).

Implement random search:
Use sklearn.model_selection.RandomizedSearchCV to perform random search.
Evaluate the model using cross-validation metrics.

Report the best hyperparameters and model performance:
Calculate and interpret regression metrics (MAE, MSE, RMSE, R-squared) for the best model.'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the Boston Housing dataset from the original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Extracting features and target
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Define a more complex neural network model using PyTorch
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Wrapper class to use PyTorch model with sklearn's RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_size, lr=0.01, epochs=10):
        self.input_size = input_size
        self.lr = lr
        self.epochs = epochs
        self.model = NeuralNet(input_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            predictions = self.model(X_tensor).squeeze()
            loss = self.loss_fn(predictions, y_tensor.squeeze())
            loss.backward()
            self.optimizer.step()

        return self

    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self.model(X_tensor).detach().numpy()
        return predictions

# Define hyperparameters for random search
param_dist = {
    'lr': [0.001, 0.01, 0.1, 0.5],
    'epochs': [50, 100, 200]
}

# Initialize PyTorchRegressor
input_size = X_scaled.shape[1]
pytorch_model = PyTorchRegressor(input_size)

# Perform random search
random_search = RandomizedSearchCV(estimator=pytorch_model, param_distributions=param_dist, cv=5, n_iter=20)
random_search.fit(X_scaled, target)

# Get the best model and hyperparameters
best_model = random_search.best_estimator_
best_lr = random_search.best_params_['lr']
best_epochs = random_search.best_params_['epochs']

# Train the best model with optimal hyperparameters
best_model.fit(X_scaled, target)

# Evaluate the best model
y_pred = best_model.predict(X_scaled)
mae = mean_absolute_error(target, y_pred)
mse = mean_squared_error(target, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(target, y_pred)

# Print results
print(f"Best learning rate: {best_lr}")
print(f"Best number of epochs: {best_epochs}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")


