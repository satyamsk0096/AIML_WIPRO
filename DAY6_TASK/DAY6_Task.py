'''
Assignment 1: Handling Missing Data and Scaling Features
Load the Titanic dataset from a CSV file.
Identify and handle missing values in the Age, Embarked, and Cabin columns using different imputation methods.
Standardize the numerical features (Age, Fare) using StandardScaler.
Normalize the numerical features using MinMaxScaler.
Compare the distributions of the scaled features using histograms.'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the Titanic dataset
file_path = r'C:\Users\sittu\OneDrive\Desktop\AIML_Wipro\DAY6_TASK\titanic.csv'  # Update this with the actual file path
titanic_df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(titanic_df.head())

# Check for missing values
print(titanic_df.isnull().sum())

# Impute missing values in Age with the mean
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)

# Impute missing values in Embarked with the mode
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

# Impute missing values in Cabin with a placeholder
titanic_df['Cabin'].fillna('Unknown', inplace=True)

# Standardize the numerical features
scaler = StandardScaler()
titanic_df[['Age_standardized', 'Fare_standardized']] = scaler.fit_transform(titanic_df[['Age', 'Fare']])

# Normalize the numerical features
minmax_scaler = MinMaxScaler()
titanic_df[['Age_normalized', 'Fare_normalized']] = minmax_scaler.fit_transform(titanic_df[['Age', 'Fare']])

# Plot histograms to compare the distributions
fig, axs = plt.subplots(3, 2, figsize=(12, 18))

# Original distributions
axs[0, 0].hist(titanic_df['Age'], bins=30, color='blue', alpha=0.7)
axs[0, 0].set_title('Original Age Distribution')
axs[0, 1].hist(titanic_df['Fare'], bins=30, color='blue', alpha=0.7)
axs[0, 1].set_title('Original Fare Distribution')

# Standardized distributions
axs[1, 0].hist(titanic_df['Age_standardized'], bins=30, color='green', alpha=0.7)
axs[1, 0].set_title('Standardized Age Distribution')
axs[1, 1].hist(titanic_df['Fare_standardized'], bins=30, color='green', alpha=0.7)
axs[1, 1].set_title('Standardized Fare Distribution')

# Normalized distributions
axs[2, 0].hist(titanic_df['Age_normalized'], bins=30, color='red', alpha=0.7)
axs[2, 0].set_title('Normalized Age Distribution')
axs[2, 1].hist(titanic_df['Fare_normalized'], bins=30, color='red', alpha=0.7)
axs[2, 1].set_title('Normalized Fare Distribution')

plt.tight_layout()
plt.show()


'''Assignment 2: Encoding Categorical Variables and Feature Engineering
Load the Iris dataset from a CSV file.
Perform one-hot encoding on the Species column.
Perform label encoding on the Species column and compare the results.
Create a new feature PetalArea by multiplying PetalLength and PetalWidth.
Create a new feature SepalArea by multiplying SepalLength and SepalWidth.'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset
file_path = r'C:\Users\sittu\OneDrive\Desktop\AIML_Wipro\DAY6_TASK\Iris.csv'  # Replace with your actual file path
iris_df = pd.read_csv(file_path)

# Perform one-hot encoding on the Species column
one_hot_encoded_df = pd.get_dummies(iris_df, columns=['Species'])

# Perform label encoding on the Species column
label_encoded_df = iris_df.copy()
le = LabelEncoder()
label_encoded_df['Species'] = le.fit_transform(label_encoded_df['Species'])

# Create new features PetalArea and SepalArea
iris_df['PetalArea'] = iris_df['PetalLength'] * iris_df['PetalWidth']
iris_df['SepalArea'] = iris_df['SepalLength'] * iris_df['SepalWidth']

# Display the results
print("Original DataFrame with new features PetalArea and SepalArea:")
print(iris_df.head())

print("\nOne-hot Encoded DataFrame:")
print(one_hot_encoded_df.head())

print("\nLabel Encoded DataFrame:")
print(label_encoded_df.head())

'''Assignment 3: Data Visualization
Load the Titanic dataset.
Create box plots to identify outliers in the Age and Fare columns.
Create histograms and KDE plots to visualize the distribution of Age and Fare.
Create scatter plots to visualize the relationship between Age and Fare, and Pclass and Survived.
Use pair plots to visualize the relationships between multiple numerical features.'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from a CSV file
titanic = pd.read_csv(r'C:\Users\sittu\OneDrive\Desktop\AIML_Wipro\DAY6_TASK\Titanic-Dataset.csv')

# Capitalize the first letter of column names
titanic.columns = [col.capitalize() for col in titanic.columns]

# Display the first few rows of the dataset to verify it loaded correctly
print(titanic.head())

# Step 1: Box Plots to Identify Outliers
plt.figure(figsize=(14, 6))

# Box plot for Age
plt.subplot(1, 2, 1)
sns.boxplot(x=titanic['Age'])
plt.title('Box Plot of Age')

# Box plot for Fare
plt.subplot(1, 2, 2)
sns.boxplot(x=titanic['Fare'])
plt.title('Box Plot of Fare')

plt.show()

# Step 2: Histograms and KDE Plots
plt.figure(figsize=(14, 12))

# Histogram and KDE for Age
plt.subplot(2, 2, 1)
sns.histplot(titanic['Age'].dropna(), kde=True)
plt.title('Histogram and KDE of Age')

# Histogram and KDE for Fare
plt.subplot(2, 2, 2)
sns.histplot(titanic['Fare'], kde=True)
plt.title('Histogram and KDE of Fare')

plt.show()

# Step 3: Scatter Plots
plt.figure(figsize=(14, 6))

# Scatter plot for Age vs Fare
plt.subplot(1, 2, 1)
sns.scatterplot(x=titanic['Age'], y=titanic['Fare'])
plt.title('Scatter Plot of Age vs Fare')

# Scatter plot for Pclass vs Survived
plt.subplot(1, 2, 2)
sns.scatterplot(x=titanic['Pclass'], y=titanic['Survived'])
plt.title('Scatter Plot of Pclass vs Survived')

plt.show()

# Step 4: Pair Plots
sns.pairplot(titanic[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')
plt.show()
