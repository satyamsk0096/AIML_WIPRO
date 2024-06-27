import numpy as np
from scipy import stats

data = [1, 5, 6, 2, 3, 4]
  #      1  3  4  5  6  7  9
data = np.array(data)

mean = np.mean(data)
print(f"Mean: {mean}")
median = np.median(data)
print(f"Median: {median}")
mode_result = stats.mode(data)
print(mode_result)
mode = mode_result[0] if len(mode_result) > 0 else None
count = mode_result[1] if len(mode_result) > 0 else None
print(f"Mode: {mode} Count: {count}")

variance = np.var(data, ddof=1)  # ddof=1 for sample variance
print(f"Variance: {variance}")

std_dev = np.std(data, ddof=1)
print(f"Standard Deviation: {std_dev}")

data_range = np.ptp(data)
print(f"Range: {data_range}")

skewness = stats.skew(data)
print(f"Skewness: {skewness}")

kurtosis = stats.kurtosis(data)
print(f"Kurtosis: {kurtosis}")

#=====================================================================

import matplotlib.pyplot as plt

# Visualization Techniques
# Histogram
plt.hist(data, bins=5, edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Box Plot
plt.boxplot(data)
plt.title('Box Plot')
plt.ylabel('Value')
plt.show()

#==========================================================================