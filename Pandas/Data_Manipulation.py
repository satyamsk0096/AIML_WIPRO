
'''
Data Manipulation with Pandas
'''

#Reading Data from CSV, Excel, JSON, and SQL

import pandas as pd

# Creating a sample DataFrame
data = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
})

data.to_csv('sample_data.csv', index=False)
data.to_excel('sample_data.xlsx', index=False)
data.to_json('sample_data.json', orient='records')


# Reading data

csv_data = pd.read_csv('sample_data.csv')
print("Data from CSV:\n", csv_data)

excel_data = pd.read_excel('sample_data.xlsx')
print("Data from Excel:\n", excel_data)

json_data = pd.read_json('sample_data.json')
print("Data from JSON:\n", json_data)


import sqlite3
# Create an in-memory SQLite database and insert the sample data
conn = sqlite3.connect(':memory:')
data.to_sql('sample_table', conn, index=False, if_exists='replace')

# Reading data from SQL
sql_data = pd.read_sql('SELECT * FROM sample_table', conn)
print("Data from SQL:\n", sql_data)



#==========================================================================
