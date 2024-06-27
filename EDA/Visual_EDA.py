import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset
iris = sns.load_dataset('iris')

# Univariate Analysis
# Histogram
plt.figure(figsize=(10, 5))
sns.histplot(iris['sepal_length'], kde=True)
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()

# Box Plot
plt.figure(figsize=(10, 5))
sns.boxplot(x=iris['sepal_length'])
plt.title('Box Plot of Sepal Length')
plt.xlabel('Sepal Length')
plt.show()

# Bivariate Analysis
# Scatter Plot
plt.figure(figsize=(10, 5))
sns.scatterplot(x='sepal_length', y='sepal_width', data=iris)
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Box Plot
plt.figure(figsize=(10, 5))
sns.boxplot(x='species', y='sepal_length', data=iris)
plt.title('Box Plot of Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length')
plt.show()

# Multivariate Analysis
# Pair Plot
sns.pairplot(iris, hue='species')
plt.title('Pair Plot of Iris Dataset')
plt.show()

# Heatmap
numeric_iris = iris.drop(columns=['species'])
correlation_matrix = numeric_iris.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Matrix')
plt.show()

# Facet Grid
g = sns.FacetGrid(iris, col='species')
g.map(sns.scatterplot, 'sepal_length', 'sepal_width')
plt.show()
