'''Question 1:

Install PyTorch and set up the environment.
Build a simple neural network with one hidden layer using PyTorch.
Implement forward propagation for the neural network.
Print the architecture and output of the network for a given input tensor.

Implement a method to monitor the training progress by plotting the loss curve.
Experiment with different learning rates and batch sizes to improve the performance of the neural network.
Implement an early stopping mechanism to halt training when the validation loss stops improving.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Define the architecture
input_size = 2  # Example input size
hidden_size = 5
output_size = 1

model = SimpleNN(input_size, hidden_size, output_size)
print(model)

# Example input tensor
input_tensor = torch.tensor([[1.0, 2.0]])
output = model(input_tensor)
print("Output:", output)

# Generate some synthetic data for training
torch.manual_seed(42)
X_train = torch.randn(100, input_size)
y_train = torch.randn(100, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# Experiment with different learning rates and batch sizes
learning_rate = 0.1
batch_size = 16

# Update the optimizer with the new learning rate
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop with mini-batch gradient descent
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Calculate and store the epoch loss
    epoch_loss = loss.item()
    losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# Split data into training and validation sets
validation_split = 0.2
split_index = int(X_train.size(0) * (1 - validation_split))
X_val, y_val = X_train[split_index:], y_train[split_index:]
X_train, y_train = X_train[:split_index], y_train[:split_index]

# Early stopping parameters
patience = 10
best_loss = float('inf')
trigger_times = 0

# Training loop with early stopping
num_epochs = 100
losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Calculate validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_losses.append(val_loss)
    
    # Check for early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping on epoch {epoch+1}')
            break
    
    # Store and print the epoch loss
    epoch_loss = loss.item()
    losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

# Plot the loss curves
plt.plot(losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.show()








print("==================================Assignment No 2================================")



