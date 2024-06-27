import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Preparation
# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Create custom dataset
class HousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader for training and validation
train_dataset = HousingDataset(X_train, y_train)
val_dataset = HousingDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 2. Neural Network Architecture
class HousingPriceNN(nn.Module):
    def __init__(self, input_dim):
        super(HousingPriceNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)  # Dropout regularization
        
    def forward(self, x):
        x = F.elu(self.fc1(x))  # Hidden layer with ELU activation
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)  # Hidden layer with Leaky ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # Output layer
        return x

input_dim = X_tensor.shape[1]
model = HousingPriceNN(input_dim)

# 3. Training and Optimization
# Loss function
criterion = nn.MSELoss()

# Optimizers
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
optimizer_sgd = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Lower learning rate for SGD

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, grad_clip=None):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if grad_clip:  # Apply gradient clipping
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

# Train with Adam
print("Training with Adam Optimizer")
train_model(model, train_loader, val_loader, criterion, optimizer_adam, epochs=50)

# Reset model parameters before training with SGD
model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

# Train with SGD with momentum
print("Training with SGD Optimizer")
train_model(model, train_loader, val_loader, criterion, optimizer_sgd, epochs=50, grad_clip=1.0)  # Apply gradient clipping

# 4. Evaluation and Visualization
def evaluate_model(model, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())
    predictions = np.concatenate(predictions).flatten()
    actuals = np.concatenate(actuals).flatten()
    return predictions, actuals

# Get predictions and actual values
pred_train, actual_train = evaluate_model(model, train_loader)
pred_val, actual_val = evaluate_model(model, val_loader)

# Calculate performance metrics
mse_train = mean_squared_error(actual_train, pred_train)
r2_train = r2_score(actual_train, pred_train)
mse_val = mean_squared_error(actual_val, pred_val)
r2_val = r2_score(actual_val, pred_val)

print(f'Training MSE: {mse_train:.4f}, R2: {r2_train:.4f}')
print(f'Validation MSE: {mse_val:.4f}, R2: {r2_val:.4f}')

# Visualization
plt.figure(figsize=(10, 5))
plt.scatter(actual_val, pred_val, label='Predicted vs Actual')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()
