'''Exercise 1: Load Datasets from Various File Formats
Load a CSV file named data.csv into a Pandas DataFrame.
Load a JSON file named data.json into a Pandas DataFrame.
Load an Excel file named data.xlsx into a Pandas DataFrame.
Ask how many rows to display from the DataFrame and display the results'''

import pandas as pd
import matplotlib.pyplot as plt


# Load CSV file into a Pandas DataFrame
csv_file_path = r'C:\Users\sittu\OneDrive\Desktop\AIML_Wipro\sample_data.csv'
data_csv = pd.read_csv(csv_file_path)

# Convert CSV to JSON
json_file_path = 'data.json'
data_csv.to_json(json_file_path, orient='records')

# Convert CSV to Excel
excel_file_path = 'data.xlsx'
data_csv.to_excel(excel_file_path, index=False)

# Load JSON file into a Pandas DataFrame
data_json = pd.read_json(json_file_path)

# Load Excel file into a Pandas DataFrame
data_excel = pd.read_excel(excel_file_path)

# Asking user for number of rows to display
num_rows = int(input("Enter the number of rows to display: "))

# Displaying specified number of rows for each DataFrame
print("\nCSV DataFrame:")
print(data_csv.head(num_rows))

print("\nJSON DataFrame:")
print(data_json.head(num_rows))

print("\nExcel DataFrame:")
print(data_excel.head(num_rows))


print('============================================Excercise No 2===================================================')


'''Exercise 2: Perform Data Cleaning and Transformation Tasks
Load a dataset with missing values.
Identify the columns with missing values.
Fill the missing values with the mean of the column.

Load a dataset where numerical columns are mistakenly read as strings.
Convert the columns to appropriate data types.

Load a dataset with duplicate rows.
Remove the duplicate rows and display the cleaned DataFrame.'''
import pandas as pd
import numpy as np

# Task 1: Load a dataset with missing values
missing_data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, np.nan],
    'C': [1, np.nan, 3, np.nan, 5]
}
missing_data_df = pd.DataFrame(missing_data)

# Identify columns with missing values
columns_with_missing_values = missing_data_df.columns[missing_data_df.isnull().any()]

# Fill missing values with mean of the column
missing_data_df[columns_with_missing_values] = missing_data_df[columns_with_missing_values].fillna(missing_data_df.mean())

# Task 2: Load a dataset where numerical columns are mistakenly read as strings
wrong_data_types = {
    'A': ['1', '2', '3', '4', '5'],
    'B': ['2.5', '3.6', '4.2', '5.1', '6.3'],
    'C': ['1.2', '2.4', '3.5', '4.7', '5.9']
}
wrong_data_types_df = pd.DataFrame(wrong_data_types)

# Convert columns to appropriate data types
wrong_data_types_df = wrong_data_types_df.astype(float)

# Task 3: Load a dataset with duplicate rows
duplicate_data = {
    'A': [1, 2, 3, 4, 4],
    'B': [5, 6, 7, 8, 8],
    'C': [9, 10, 11, 12, 12]
}
duplicate_data_df = pd.DataFrame(duplicate_data)

# Add some duplicate rows
duplicate_data_df = pd.concat([duplicate_data_df, duplicate_data_df.iloc[2:4]], ignore_index=True)

# Remove duplicate rows
cleaned_df = duplicate_data_df.drop_duplicates()

# Display cleaned DataFrame
print("Missing Data DataFrame:")
print(missing_data_df)
print("\nWrong Data Types DataFrame:")
print(wrong_data_types_df)
print("\nDuplicate Data DataFrame:")
print(cleaned_df)

print("==========================Excerice No 3======================================================================")

'''Exercise 3: Merge and Join Multiple DataFrames
Load two DataFrames, df1 and df2, with a common column id.
Perform an inner join on the id column.
Display the merged DataFrame.

Load two DataFrames, df1 and df2, with a common column id.
Perform an outer join on the id column.
Display the merged DataFrame.

Load two DataFrames with the same columns.
Concatenate the DataFrames vertically.
Display the concatenated DataFrame.'''


#Excercise 3
# Task 1: Inner join two DataFrames
df1 = pd.DataFrame({'id': [1, 2, 3, 4],
                    'value1': ['A', 'B', 'C', 'D']})
df2 = pd.DataFrame({'id': [2, 3, 4, 5],
                    'value2': ['X', 'Y', 'Z', 'W']})

inner_merged_df = pd.merge(df1, df2, on='id', how='inner')

# Display the merged DataFrame
print("Inner Merged DataFrame:")
print(inner_merged_df)

# Task 2: Outer join two DataFrames
outer_merged_df = pd.merge(df1, df2, on='id', how='outer')

# Display the merged DataFrame
print("\nOuter Merged DataFrame:")
print(outer_merged_df)

# Task 3: Concatenate two DataFrames vertically
df3 = pd.DataFrame({'id': [6, 7, 8, 9],
                    'value1': ['E', 'F', 'G', 'H']})

concatenated_df = pd.concat([df1, df3], ignore_index=True)

# Display the concatenated DataFrame
print("\nConcatenated DataFrame:")
print(concatenated_df)


print("===========================Excercise No 4================================================")


'''Exercise 4: Group and Aggregate Data for Analysis

Load a dataset with columns category and value.
Group the data by category and compute the sum of value for each category.
Display the aggregated DataFrame.

Load a dataset with columns category and value.
Group the data by category and compute the mean and standard deviation of value for each category.
Display the aggregated DataFrame.

Load a dataset with columns category, subcategory, and value.
Create a pivot table that shows the sum of value for each combination of category and subcategory.
Display the pivot table.'''


import pandas as pd

# Task 1: Group by category and compute sum of value
data1 = {'category': ['A', 'B', 'A', 'B', 'A'],
         'value': [10, 20, 30, 40, 50]}
df1 = pd.DataFrame(data1)

aggregated_df1 = df1.groupby('category').agg({'value': 'sum'})

# Display the aggregated DataFrame
print("Aggregated DataFrame (Sum of values by category):")
print(aggregated_df1)

# Task 2: Group by category and compute mean and standard deviation of value
data2 = {'category': ['A', 'B', 'A', 'B', 'A'],
         'value': [10, 20, 30, 40, 50]}
df2 = pd.DataFrame(data2)

aggregated_df2 = df2.groupby('category').agg({'value': ['mean', 'std']})

# Display the aggregated DataFrame
print("\nAggregated DataFrame (Mean and Standard Deviation of values by category):")
print(aggregated_df2)

# Task 3: Create pivot table showing sum of value for each combination of category and subcategory
data3 = {'category': ['A', 'A', 'B', 'B', 'A'],
         'subcategory': ['X', 'Y', 'X', 'Y', 'X'],
         'value': [10, 20, 30, 40, 50]}
df3 = pd.DataFrame(data3)

pivot_table_df = df3.pivot_table(index='category', columns='subcategory', values='value', aggfunc='sum')

# Display the pivot table
print("\nPivot Table (Sum of values for each combination of category and subcategory):")
print(pivot_table_df)

print("===================Exercise No 5================================================")

'''Exercise 5: Create Basic Visualizations to Explore Data

Load a time series dataset.
Create a line plot to visualize the trend over time.
Display the plot.
'''
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# Sample time series data
time_series_data = """date,value
2023-01-01,100
2023-01-02,110
2023-01-03,105
2023-01-04,115
2023-01-05,120
"""
time_series_df = pd.read_csv(StringIO(time_series_data), parse_dates=['date'])

# Create a line plot to visualize the trend over time
plt.figure(figsize=(10, 6))
plt.plot(time_series_df['date'], time_series_df['value'], marker='o')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Trend Over Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




