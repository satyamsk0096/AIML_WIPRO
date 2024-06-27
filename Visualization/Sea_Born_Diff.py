import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
iris = sns.load_dataset('iris')

# Create a pair plot
sns.pairplot(iris, hue='species')
plt.title('Pair Plot of Iris Dataset')
plt.show()

# Create a joint plot
sns.jointplot(x='sepal_length', y='sepal_width', data=iris, kind='scatter', hue='species')
plt.suptitle('Joint Plot of Sepal Length vs Sepal Width')
plt.show()

# Exclude non-numeric columns
numeric_iris = iris.drop(columns=['species'])

# Create a sample correlation matrix
data = numeric_iris.corr()

# Create a heatmap
sns.heatmap(data, annot=True, cmap='coolwarm')
plt.title('Heatmap of Iris Correlation Matrix')
plt.show()

# Create a facet grid
g = sns.FacetGrid(iris, col='species')
g.map(sns.histplot, 'sepal_length')
plt.suptitle('Facet Grid of Sepal Length by Species')
plt.show()

#Customising Plots

# Set a theme
sns.set_theme(style='whitegrid')

# Create a scatter plot with the theme
sns.scatterplot(x='sepal_length', y='sepal_width', data=iris, hue='species')
plt.title('Scatter Plot with Whitegrid Theme')
plt.show()

# Set a color palette
sns.set_palette('pastel')

# Create a bar plot with the color palette
sns.barplot(x='species', y='sepal_length', data=iris)
plt.title('Bar Plot with Pastel Color Palette')
plt.show()


# Create a bar plot with annotations
sns.barplot(x='species', y='sepal_length', data=iris, errorbar='sd')

# Add annotations
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.title('Bar Plot with Statistical Annotations')
plt.show()


# Create a scatter plot with Seaborn
sns.scatterplot(x='sepal_length', y='sepal_width', data=iris, hue='species')

# Customize with Matplotlib
plt.title('Scatter Plot with Matplotlib Customization')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend(title='Species')
plt.grid(True)
plt.show()
