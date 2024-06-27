'''Assignment 1: Implementing Linear Regression with PyTorch

Problem 1: Load a dataset (such as the Boston Housing dataset) and implement a simple linear regression model to predict housing prices. Use Mean Squared Error (MSE) as the loss function.
Problem 2: Plot the predicted vs actual values for the training and test sets.'''

# Problem 1
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load the California Housing dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

input_dim = X_train.shape[1]
model = LinearRegressionModel(input_dim)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    train_predictions = model(X_train_tensor).numpy()
    test_predictions = model(X_test_tensor).numpy()

# Calculate and print the MSE for training and testing sets
train_mse = np.mean((train_predictions - y_train_tensor.numpy()) ** 2)
test_mse = np.mean((test_predictions - y_test_tensor.numpy()) ** 2)

print(f'Training MSE: {train_mse:.4f}')
print(f'Testing MSE: {test_mse:.4f}')

# Problem 2

# Plot for training set
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, train_predictions, alpha=0.7)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Training Set: Actual vs Predicted Prices')

# Plot for testing set
plt.subplot(1, 2, 2)
plt.scatter(y_test, test_predictions, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Testing Set: Actual vs Predicted Prices')

plt.show()

'''Assignment 2: Performing Polynomial Regression and Feature Engineering

Problem 1: Load a dataset and perform polynomial regression. Add polynomial features up to degree 3 and train the model using these features.
Problem 2: Compare the performance of the polynomial regression model with the linear regression model using MSE.'''

# Problem 1

from sklearn.preprocessing import PolynomialFeatures

# Add polynomial features up to degree 3
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Split the dataset into training and testing sets
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_poly_tensor = torch.tensor(X_train_poly, dtype=torch.float32)
y_train_poly_tensor = torch.tensor(y_train_poly, dtype=torch.float32).view(-1, 1)
X_test_poly_tensor = torch.tensor(X_test_poly, dtype=torch.float32)
y_test_poly_tensor = torch.tensor(y_test_poly, dtype=torch.float32).view(-1, 1)

# Define the Linear Regression model for polynomial regression
poly_model = LinearRegressionModel(X_train_poly.shape[1])

# Define the loss function and the optimizer
poly_criterion = nn.MSELoss()
poly_optimizer = optim.SGD(poly_model.parameters(), lr=0.01)

# Train the polynomial regression model
for epoch in range(num_epochs):
    poly_model.train()
    
    # Forward pass
    poly_outputs = poly_model(X_train_poly_tensor)
    poly_loss = poly_criterion(poly_outputs, y_train_poly_tensor)
    
    # Backward pass and optimization
    poly_optimizer.zero_grad()
    poly_loss.backward()
    poly_optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {poly_loss.item():.4f}')

# Evaluate the polynomial model
poly_model.eval()
with torch.no_grad():
    train_poly_predictions = poly_model(X_train_poly_tensor).numpy()
    test_poly_predictions = poly_model(X_test_poly_tensor).numpy()

# Calculate and print the MSE for training and testing sets for polynomial regression
train_poly_mse = np.mean((train_poly_predictions - y_train_poly_tensor.numpy()) ** 2)
test_poly_mse = np.mean((test_poly_predictions - y_test_poly_tensor.numpy()) ** 2)

print(f'Training MSE (Polynomial): {train_poly_mse:.4f}')
print(f'Testing MSE (Polynomial): {test_poly_mse:.4f}')

# Problem 2

print(f'Linear Regression Training MSE: {train_mse:.4f}')
print(f'Linear Regression Testing MSE: {test_mse:.4f}')
print(f'Polynomial Regression Training MSE: {train_poly_mse:.4f}')
print(f'Polynomial Regression Testing MSE: {test_poly_mse:.4f}')

'''Assignment 3: Applying Ridge and Lasso Regression

Problem 1: Implement Ridge regression on the same dataset used in Assignment 1. Use cross-validation to select the best regularization parameter (alpha).
Problem 2: Implement Lasso regression on the same dataset. Use cross-validation to select the best regularization parameter (alpha).
Problem 3: Compare the performance of Ridge, Lasso, and standard linear regression models in terms of MSE and interpret the results.'''

# Problem 1
from sklearn.linear_model import RidgeCV

# Define the range of alphas to test
alphas = np.logspace(-6, 6, 13)

# Implement Ridge regression with cross-validation
ridge_model = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_model.fit(X_train, y_train)

# Get the best alpha value
best_alpha_ridge = ridge_model.alpha_
print(f'Best alpha for Ridge regression: {best_alpha_ridge}')

# Predict on training and testing sets
ridge_train_predictions = ridge_model.predict(X_train)
ridge_test_predictions = ridge_model.predict(X_test)

# Calculate and print the MSE for training and testing sets
ridge_train_mse = np.mean((ridge_train_predictions - y_train) ** 2)
ridge_test_mse = np.mean((ridge_test_predictions - y_test) ** 2)

print(f'Ridge Regression Training MSE: {ridge_train_mse:.4f}')
print(f'Ridge Regression Testing MSE: {ridge_test_mse:.4f}')

# Problem 2

from sklearn.linear_model import LassoCV

# Implement Lasso regression with cross-validation
lasso_model = LassoCV(alphas=alphas, max_iter=10000, cv=5)
lasso_model.fit(X_train, y_train)

# Get the best alpha value
best_alpha_lasso = lasso_model.alpha_
print(f'Best alpha for Lasso regression: {best_alpha_lasso}')

# Predict on training and testing sets
lasso_train_predictions = lasso_model.predict(X_train)
lasso_test_predictions = lasso_model.predict(X_test)

# Calculate and print the MSE for training and testing sets
lasso_train_mse = np.mean((lasso_train_predictions - y_train) ** 2)
lasso_test_mse = np.mean((lasso_test_predictions - y_test) ** 2)

print(f'Lasso Regression Training MSE: {lasso_train_mse:.4f}')
print(f'Lasso Regression Testing MSE: {lasso_test_mse:.4f}')

# Problem 3
print(f'Linear Regression Training MSE: {train_mse:.4f}')
print(f'Linear Regression Testing MSE: {test_mse:.4f}')
print(f'Polynomial Regression Training MSE: {train_poly_mse:.4f}')
print(f'Polynomial Regression Testing MSE: {test_poly_mse:.4f}')
print(f'Ridge Regression Training MSE: {ridge_train_mse:.4f}')
print(f'Ridge Regression Testing MSE: {ridge_test_mse:.4f}')
print(f'Lasso Regression Training MSE: {lasso_train_mse:.4f}')
print(f'Lasso Regression Testing MSE: {lasso_test_mse:.4f}')
