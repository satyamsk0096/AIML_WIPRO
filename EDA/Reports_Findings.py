import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

# 2. Data Description
summary_stats = iris.describe()
print("Summary Statistics:")
print(summary_stats)

# 3. Check for missing values
missing_values = iris.isnull().sum()
print("Missing Values:")
print(missing_values)

# 4. Data Visualization
# Pairplot
sns.pairplot(iris, hue='species')
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris)
plt.title('Box Plot of Iris Features')
plt.show()

# Heatmap
numeric_iris = iris.drop(columns=['species'])
correlation_matrix = numeric_iris.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Matrix')
plt.show()

# 5. Insights and Observations
print("Insights and Observations:")
print("- Species Differentiation: The pairplot indicates clear separation between the species based on petal length and width.")
print("- Outliers: The box plot reveals some outliers in sepal width.")
print("- Correlation: The heatmap shows high correlation between petal length and petal width.")

# 6. Conclusions and Next Steps
print("Conclusions and Next Steps:")
print("- Conclusions: The EDA shows that petal dimensions are good indicators of species differentiation. Sepal dimensions are less distinctive.")
print("- Next Steps: Further analysis could include modeling to predict species based on the measurements, and investigating the outliers in more detail.")