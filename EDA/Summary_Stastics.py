import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset
iris = sns.load_dataset('iris')

# 4.2.1 Descriptive Statistics
mean_sepal_length = iris['sepal_length'].mean()
median_sepal_length = iris['sepal_length'].median()
mode_sepal_length = iris['sepal_length'].mode()[0]

print(f"Mean Sepal Length: {mean_sepal_length}")
print(f"Median Sepal Length: {median_sepal_length}")
print(f"Mode Sepal Length: {mode_sepal_length}")

# 4.2.2 Dispersion Statistics
variance_sepal_length = iris['sepal_length'].var()
std_dev_sepal_length = iris['sepal_length'].std()

print(f"Variance of Sepal Length: {variance_sepal_length}")
print(f"Standard Deviation of Sepal Length: {std_dev_sepal_length}")

# 4.2.3 Correlation Analysis
import seaborn as sns

iris = sns.load_dataset('iris')

# Drop the non-numeric 'species' column
numeric_iris = iris.drop(columns=['species'])

# Calculate the correlation matrix
correlation_matrix = numeric_iris.corr()

print("Correlation matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Matrix')
plt.show()