import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a sample DataFrame
data = pd.DataFrame({
    'x': np.random.rand(100),
    'y': np.random.rand(100)
})

# Create a scatterplot
sns.scatterplot(x='x', y='y', data=data)
plt.title('Scatterplot of x vs y')
plt.show()

# Create a sample DataFrame with a time series
data = pd.DataFrame({
    'time': pd.date_range(start='1/1/2020', periods=100),
    'value': np.random.rand(100).cumsum()
})

# Create a lineplot
sns.lineplot(x='time', y='value', data=data)
plt.title('Lineplot of Value over Time')
plt.show()

# Create a sample DataFrame
data = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D'],
    'value': [10, 20, 15, 25]
})

# Create a barplot
sns.barplot(x='category', y='value', data=data)
plt.title('Barplot of Categories')
plt.show()

# Create a sample DataFrame
data = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'C', 'C']
})

# Create a countplot
sns.countplot(x='category', data=data)
plt.title('Countplot of Categories')
plt.show()

# Create a sample DataFrame
data = pd.DataFrame({
    'value': np.random.randn(100)
})

# Create a histogram
sns.histplot(data['value'], bins=10)
plt.title('Histogram of Values')
plt.show()


# Create a KDE plot
sns.kdeplot(data['value'])
plt.title('KDE Plot of Values')
plt.show()