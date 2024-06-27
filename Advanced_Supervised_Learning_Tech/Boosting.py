# GBM classifier

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Define a simple neural network as a weak learner
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Train a weak learner
def train_model(model, X_train, y_train, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model


# Train multiple weak learners sequentially
num_models = 3
learning_rate = 0.1
models = []
residuals = y_train.clone()

for _ in range(num_models):
    model = Net()
    model = train_model(model, X_train, residuals)
    models.append(model)

    # Predict on the training data
    predictions = model(X_train).detach()

    # Update residuals
    residuals -= learning_rate * predictions
    #print(f'Residuals :  {residuals}')

# Function to make predictions with the GBM model
def predict(models, X, learning_rate):
    predictions = torch.zeros((X.shape[0], 1))
    for model in models:
        predictions += learning_rate * model(X).detach()
       # print(f'Pred : {predictions}')
    return predictions


# Make predictions on the test set
y_pred = predict(models, X_test, learning_rate)

# Evaluate the GBM model
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
r2 = r2_score(y_test.numpy(), y_pred.numpy())

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# Create AdaBoost with Logistic Regression as the base estimator
log_reg_boost = AdaBoostClassifier(estimator=LogisticRegression(max_iter=10000),
                                   n_estimators=50, random_state=42)

# Train the model
log_reg_boost.fit(X_train, y_train)

# Make predictions
log_reg_preds = log_reg_boost.predict(X_test)

# Evaluate the model
accuracy_lr = accuracy_score(y_test, log_reg_preds)
precision_lr = precision_score(y_test, log_reg_preds)
recall_lr = recall_score(y_test, log_reg_preds)
f1_lr = f1_score(y_test, log_reg_preds)

print("Logistic Regression Boosting:")
print(f"Accuracy: {accuracy_lr}, Precision: {precision_lr}, Recall: {recall_lr}, F1 Score: {f1_lr}")


from sklearn.tree import DecisionTreeClassifier

# Create AdaBoost with Decision Tree as the base estimator
dt_boost = AdaBoostClassifier(estimator=DecisionTreeClassifier(),
                              n_estimators=50, random_state=42)

# Train the model
dt_boost.fit(X_train, y_train)

# Make predictions
dt_preds = dt_boost.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, dt_preds)
precision_dt = precision_score(y_test, dt_preds)
recall_dt = recall_score(y_test, dt_preds)
f1_dt = f1_score(y_test, dt_preds)

print("\nDecision Tree Boosting:")
print(f"Accuracy: {accuracy_dt}, Precision: {precision_dt}, Recall: {recall_dt}, F1 Score: {f1_dt}")