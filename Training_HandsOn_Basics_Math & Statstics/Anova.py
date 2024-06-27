
import pandas as pd
from scipy import stats

'''

Test scores from three different teaching methods.
One-way ANOVA test
'''

'''data = {
    'Method A': [85, 87, 88, 94, 78],
    'Method B': [80, 85, 84, 89, 82],
    'Method C': [78, 82, 83, 88, 90]
}
'''

data = {
    'Method A': [55, 67, 58, 64, 67],
    'Method B': [80, 85, 84, 89, 82],
    'Method C': [78, 82, 83, 88, 90]
}

# Convert data to a DataFrame
df = pd.DataFrame(data)
print(df)

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(df['Method A'],
                                      df['Method B'], df['Method C'])

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

# Significance level
alpha = 0.05

# Decision
if p_value < alpha:
    print("Reject the null hypothesis: At least one teaching method is significantly different.")
else:
    print("Fail to reject the null hypothesis: No significant difference between teaching methods.")

#==================================================================================

import pandas as pd
from scipy import stats

# Sample data: Heights of plants (in cm) under 4 different fertilizers
data = {
    'Fertilizer A': [22, 24, 21, 23, 26],
    'Fertilizer B': [25, 27, 28, 24, 23],
    'Fertilizer C': [20, 21, 19, 22, 18],
    'Fertilizer D': [27, 21, 21, 22, 28]
}

# Convert data to a DataFrame
df = pd.DataFrame(data)
print(df)

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(df['Fertilizer A'],
                    df['Fertilizer B'], df['Fertilizer C'],
                    df['Fertilizer D'])

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

# Significance level
alpha = 0.05

# Decision
if p_value < alpha:
    print("Reject the null hypothesis: At least one fertilizer is significantly different.")
else:
    print("Fail to reject the null hypothesis: No significant difference between fertilizers.")

#===============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Sample data: Heights of plants (in cm) under three different fertilizers
data = {
    'Fertilizer': np.repeat(['A', 'B', 'C'], 5),
    'Height': [22, 24, 21, 23, 26, 25, 27, 28, 24, 23, 20, 21, 19, 22, 18]
}

# Convert data to a DataFrame
df = pd.DataFrame(data)
print(df)

# Perform Tukey's HSD post-hoc test
tukey_result = pairwise_tukeyhsd(endog=df['Height'],
                                 groups=df['Fertilizer'], alpha=0.05)

print(tukey_result)

# Plot the results
tukey_result.plot_simultaneous()

#============================================================
