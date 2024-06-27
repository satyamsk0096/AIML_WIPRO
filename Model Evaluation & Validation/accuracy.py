# Calculating Accuracy
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define a simple neural network model for classification
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


# Initialize model, loss function, and optimizer
model = SimpleNN(input_size=X_tensor.shape[1], num_classes=len(torch.unique(y_tensor)))
criterion = torch.nn.CrossEntropyLoss()
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
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
y_pred = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
y_pred = torch.tensor(y_pred)


# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    correct = (y_true == y_pred).float()
    accuracy = correct.sum() / len(correct)
    return accuracy.item()

# Calculate accuracy
accuracy = calculate_accuracy(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')


# Precision, Recall, and F1-Score

from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1-score with zero_division parameter
precision = precision_score(y_test.numpy(), y_pred.numpy(), average='macro', zero_division=0)
recall = recall_score(y_test.numpy(), y_pred.numpy(), average='macro', zero_division=0)
f1 = f1_score(y_test.numpy(), y_pred.numpy(), average='macro', zero_division=0)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')


# ROC Curve and AUC

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Calculate probabilities
model.eval()
y_prob = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        y_prob.extend(probabilities.numpy())
y_prob = np.array(y_prob)  # Convert list of NumPy arrays to a single NumPy array
y_prob = torch.from_numpy(y_prob)  # Convert the NumPy array to a PyTorch tensor

# Binarize the output for ROC
y_test_binarized = torch.nn.functional.one_hot(y_test).numpy()
y_prob = y_prob.numpy()

# Plot ROC curve and calculate AUC for each class
plt.figure(figsize=(10, 8))
for i in range(len(data.target_names)):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {data.target_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

