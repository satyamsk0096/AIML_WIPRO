import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load the California housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor,
                                                    test_size=0.2, random_state=42)

# Define a simple neural network model for regression
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

# Function to calculate Mean Absolute Error (MAE)
def calculate_mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

# Function to calculate R-squared (R²)
def calculate_r2(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
    ss_residual = torch.sum((y_true - y_pred) ** 2).item()
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Initialize model, loss function, and optimizer
model = SimpleNN(input_size=X_tensor.shape[1])
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Train the model
model.train()
for epoch in range(100):  # Train for 100 epochs
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))  # Ensure labels are reshaped properly
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
y_pred = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        y_pred.append(outputs)
y_pred = torch.cat(y_pred, dim=0)

# Reshape y_pred to match y_test if necessary (depends on how your batches were processed)
if y_pred.shape != y_test.shape:
    y_pred = y_pred.view(y_test.shape)

# Calculate metrics: MAE, MSE, RMSE
mae = calculate_mae(y_test, y_pred)
mse = torch.mean((y_test - y_pred) ** 2).item()
rmse = torch.sqrt(torch.mean((y_test - y_pred) ** 2)).item()

# Calculate R-squared (R²)
r2 = calculate_r2(y_test, y_pred)

# Print results
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R-squared (R²): {r2:.4f}')
