import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, delimiter=';')

# Preprocess the data: normalize the features
X = data.drop('quality', axis=1)
y = data['quality'].apply(lambda x: 1 if x >= 7 else 0)  # Binarize the target variable

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Function to implement k-NN and evaluate performance
def evaluate_knn(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Experiment with different values of k
k_values = [3, 5, 7]
accuracies = {k: evaluate_knn(k) for k in k_values}

# Report the accuracy for each value of k
for k in k_values:
    print(f"Accuracy for k={k}: {accuracies[k]:.4f}")
