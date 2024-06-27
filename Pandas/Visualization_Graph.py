#Visualization with Pandas
#Basic plotting with Pandas

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Creating a sample DataFrame
df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])

# Plotting
df.plot()
plt.title('Basic Plot')
plt.show()

# Customizing plot with labels and title
df.plot()
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Customized Plot')
plt.show()

#Integration with Matplotlib and Seaborn
import seaborn as sns

# Using Seaborn for a more advanced plot
sns.lineplot(data=df)
plt.title('Seaborn Line Plot')
plt.show()