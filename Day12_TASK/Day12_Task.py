'''DAY 12

Exercise 1: Implement PCA on a Dataset using PyTorch and Visualize the Results
Dataset: Wine Quality Dataset

Objective: Implement PCA on the Wine Quality dataset to reduce its dimensionality and visualize the results in 2D.

Steps:
Load the Wine Quality dataset:
Use pandas to load the Wine Quality dataset from the UCI Machine Learning Repository.

Preprocess the data:
Normalize the features using StandardScaler.

Implement PCA using PyTorch:
Compute the covariance matrix.
Compute eigenvalues and eigenvectors of the covariance matrix.
Project the data onto the first two principal components.

Visualize the Results:
Use matplotlib to plot the 2D projection of the data.'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt

# Step 1: Load the Wine Quality Dataset
# Load the dataset from UCI ML Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_data = pd.read_csv(url, sep=';')

# Display the first few rows of the dataset
print(wine_data.head())

# Step 2: Preprocess the Data
# Separate features from the target variable
X = wine_data.drop('quality', axis=1)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Implement PCA using PyTorch
# Convert numpy array to PyTorch tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float)

# Compute the covariance matrix manually
mean = torch.mean(X_tensor, dim=0)
X_centered = X_tensor - mean
cov_matrix = torch.matmul(X_centered.t(), X_centered) / (X_centered.size(0) - 1)

# Compute eigenvalues and eigenvectors using torch.linalg.eigh
eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

# Sort eigenvalues and corresponding eigenvectors in descending order
eigenvalues = eigenvalues.flip(0)
eigenvectors = eigenvectors.flip(1)

# Project the data onto the first two principal components
principal_components = torch.matmul(X_centered, eigenvectors[:, :2])

# Step 4: Visualize the Results
# Convert principal components back to numpy array for plotting
principal_components_np = principal_components.detach().numpy()

# Plotting the 2D projection
plt.figure(figsize=(8, 6))
plt.scatter(principal_components_np[:, 0], principal_components_np[:, 1], c=wine_data['quality'], cmap='viridis', alpha=0.5)
plt.title('PCA on Wine Quality Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Quality')
plt.grid(True)
plt.show()



'''Exercise 2: Apply t-SNE for Dimensionality Reduction and Visualize High-Dimensional Data in 2D or 3D
Dataset: Wine Quality Dataset

Objective: Apply t-SNE on the Wine Quality dataset to reduce its dimensionality and visualize the results in 2D or 3D.

Steps:
Load the Wine Quality dataset:

Use pandas to load the Wine Quality dataset from the UCI Machine Learning Repository.

Preprocess the data:
Normalize the features using StandardScaler.

Apply t-SNE:
Use sklearn.manifold.TSNE to reduce the data to 2 or 3 dimensions.

Visualize the Results:
Use matplotlib to plot the 2D or 3D projection of the data.'''


import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load the Wine Quality dataset
wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

# Display the first few rows of the dataframe to understand its structure
print(wine_df.head())

# Step 2: Preprocess the data - Normalize the features using StandardScaler
scaler = StandardScaler()
wine_scaled = scaler.fit_transform(wine_df)

# Convert back to DataFrame for convenience
wine_scaled_df = pd.DataFrame(data=wine_scaled, columns=wine.feature_names)

# Step 3: Apply t-SNE to reduce to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
wine_tsne = tsne.fit_transform(wine_scaled_df)

# Step 4: Visualize the Results in 2D
plt.figure(figsize=(8, 6))
plt.scatter(wine_tsne[:, 0], wine_tsne[:, 1], c=wine.target, cmap='viridis')
plt.colorbar(label='Wine Class')
plt.title('t-SNE Projection of Wine Quality Dataset (2D)')
plt.xlabel('First t-SNE Component')
plt.ylabel('Second t-SNE Component')
plt.show()

# Step 5: Optional - Visualize the Results in 3D
# Apply t-SNE to reduce to 3 dimensions
tsne_3d = TSNE(n_components=3, random_state=42)
wine_tsne_3d = tsne_3d.fit_transform(wine_scaled_df)

# Plotting the 3D t-SNE projection
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(wine_tsne_3d[:, 0], wine_tsne_3d[:, 1], wine_tsne_3d[:, 2], c=wine.target, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Wine Class")
ax.add_artist(legend1)
ax.set_title('t-SNE Projection of Wine Quality Dataset (3D)')
ax.set_xlabel('First t-SNE Component')
ax.set_ylabel('Second t-SNE Component')
ax.set_zlabel('Third t-SNE Component')
plt.show()
