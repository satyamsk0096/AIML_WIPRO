'''Exercise 1: Implement k-Means Clustering on a Dataset using PyTorch
Dataset: Iris Dataset

Objective: Implement k-Means clustering on the Iris dataset to group the data into clusters. Visualize the results and evaluate the clustering performance using a suitable metric such as silhouette score.

Steps:
Load the Iris dataset:

Use pandas to load the Iris dataset from the UCI Machine Learning Repository.

Preprocess the data:
Normalize the features using StandardScaler.

Implement k-Means Clustering using PyTorch:
Initialize cluster centroids randomly.
Assign each data point to the nearest centroid.
Update centroids by computing the mean of the assigned points.
Repeat the process until convergence.

Visualize the Clusters:
Use matplotlib to plot the clusters.
If the data has more than 2 features, use PCA to reduce dimensionality before plotting.

Evaluate the Clustering Performance:
Use silhouette score from sklearn.metrics.'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load the Iris dataset from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal length", "sepal width", "petal length", "petal width", "target"]
df = pd.read_csv(url, header=None, names=column_names)
print("First five rows of the Iris dataset:")
print(df.head())

# Convert the target column to numerical values
df['target'] = df['target'].astype('category').cat.codes

# Step 2: Preprocess the data
# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns=['target']))

# Convert to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

# Step 3: Implement k-Means Clustering using PyTorch
def kmeans(X, n_clusters, n_iters=100):
    # Randomly initialize cluster centroids
    centroids = X[torch.randperm(X.size(0))[:n_clusters]]
    
    for _ in range(n_iters):
        # Compute distances between points and centroids
        distances = torch.cdist(X, centroids)
        
        # Assign each point to the nearest centroid
        cluster_assignments = torch.argmin(distances, dim=1)
        
        # Update centroids
        new_centroids = torch.stack([X[cluster_assignments == k].mean(dim=0) for k in range(n_clusters)])
        
        # Check for convergence
        if torch.allclose(new_centroids, centroids):
            break
        
        centroids = new_centroids
    
    return cluster_assignments, centroids

# Number of clusters
n_clusters = 3
# Perform k-Means clustering
cluster_assignments, centroids = kmeans(X_tensor, n_clusters)

# Step 4: Visualize the Clusters
# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the clusters
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(X_pca[cluster_assignments == i, 0], X_pca[cluster_assignments == i, 1], label=f'Cluster {i+1}')
plt.legend()
plt.title('k-Means Clustering on Iris Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Step 5: Evaluate the Clustering Performance
# Compute silhouette score
silhouette_avg = silhouette_score(X, cluster_assignments.numpy())
print(f'Silhouette Score for k-Means Clustering on Iris Dataset: {silhouette_avg}')


'''

Exercise 2: Apply Hierarchical Clustering and Create a Dendrogram

Dataset: Wine Quality Dataset

Objective: Apply hierarchical clustering on the Wine Quality dataset to group the data into clusters. Create a dendrogram to visualize the hierarchical relationships.

Steps:

Load the Wine Quality dataset:
Use pandas to load the Wine Quality dataset from the UCI Machine Learning Repository.

Preprocess the data:
Normalize the features using StandardScaler.

Apply Hierarchical Clustering:
Use scipy.cluster.hierarchy to perform hierarchical clustering.
Use linkage method to compute the hierarchical clustering.

Create and Visualize the Dendrogram:
Use dendrogram function to create the dendrogram.

Evaluate the Clustering Performance:
Use silhouette score from sklearn.metrics.'''


import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Step 1: Load the Wine Quality dataset
wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_df = pd.read_csv(wine_url, sep=';')
print("First five rows of the Wine Quality dataset:")
print(wine_df.head())

# Step 2: Preprocess the data
# Normalize the features excluding 'quality'
scaler = StandardScaler()
X_wine = scaler.fit_transform(wine_df.drop('quality', axis=1))

# Convert to DataFrame for compatibility with clustering functions
X_wine_df = pd.DataFrame(X_wine, columns=wine_df.columns[:-1])

# Step 3: Apply Hierarchical Clustering
# Perform hierarchical clustering using the 'ward' linkage method
Z = linkage(X_wine_df, method='ward')

# Step 4: Create and Visualize the Dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram (Wine Quality Dataset)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Step 5: Evaluate the Clustering Performance
# Apply Agglomerative Clustering to assign clusters for silhouette score calculation
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
cluster_assignments_wine = agg_clustering.fit_predict(X_wine_df)

# Compute silhouette score
silhouette_avg_wine = silhouette_score(X_wine_df, cluster_assignments_wine)
print(f'Silhouette Score for Hierarchical Clustering on Wine Quality Dataset: {silhouette_avg_wine}')
