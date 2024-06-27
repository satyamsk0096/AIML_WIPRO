
'''
Panel Data
Panel data, also known as longitudinal data or cross-sectional time series data, 
involves observations of multiple phenomena obtained over multiple time periods 
for the same firms or individuals. In pandas, Panel data used to be handled using 
the Panel class, but it has since been deprecated in favor of using multi-index DataFrames.

Panel Data Structure
Panel data structure allows for the storage and manipulation of three-dimensional data,
 typically with dimensions (items, major_axis, minor_axis):

Items: Axis 0, each item corresponds to a DataFrame (like different variables).
Major_axis: Axis 1, usually represents time.
Minor_axis: Axis 2, represents individual entities (like different firms or individuals).
Due to the deprecation of Panel, we now use multi-index DataFrames to handle panel data.

Creating and Manipulating Panels
Creating Panel-like Data with Multi-index DataFrames
'''

import pandas as pd
import numpy as np

# Create a multi-index DataFrame
arrays = [
    ['A', 'A', 'B', 'B'],
    [1, 2, 1, 2]
]

index = pd.MultiIndex.from_arrays(arrays, names=('person', 'time'))
data = pd.DataFrame(np.random.randn(4, 3), index=index, columns=['entity1',
                                                                 'entity2', 'entity3'])

print("Multi-index DataFrame:")
print(data)

#Manipulating Multi-index DataFrames
# Access data for variable 'A'
print("\nData for variable 'A':")
print(data.loc['A'])

# Access data for time period 1
print("\nData for time period 1:")
print(data.xs(1, level='time'))

# Adding a new row for a new time period
new_data = pd.DataFrame({
    'entity1': [0.5, 0.3],
    'entity2': [1.5, 1.3],
    'entity3': [2.5, 2.3]
}, index=pd.MultiIndex.from_product([['A', 'B'], [3]], names=['variable', 'time']))
print(new_data)

data = pd.concat([data, new_data])
print("\nData after adding new time period:")
print(data)

'''
Applications and Use Cases

Economics and Finance: Analyzing the financial performance of different firms over time.
Healthcare: Monitoring patient health metrics across different time periods.
Social Sciences: Studying the behavior of individuals across various time points.
Marketing: Observing the impact of marketing campaigns over time.

'''
 