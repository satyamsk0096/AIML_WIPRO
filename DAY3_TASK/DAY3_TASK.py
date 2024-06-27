'''Day 3

Question 1: One-Sample t-Test

Perform a one-sample t-test to determine if the sample mean is significantly different from a known population mean.

Generate a sample dataset of 30 random values from a normal distribution with a mean of 60 and a standard deviation of 10.
Perform a one-sample t-test to check if the sample mean is significantly different from 50.

'''
import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Question 1: One-Sample t-Test
# Generate a sample dataset of 30 random values from a normal distribution with a mean of 60 and a standard deviation of 10.
np.random.seed(0)
sample = np.random.normal(60, 10, 30)

# Perform a one-sample t-test to check if the sample mean is significantly different from 50.
t_stat, p_value = stats.ttest_1samp(sample, 50)
print(f"One-Sample t-Test:\nT-statistic: {t_stat:.4f}, P-value: {p_value:.4f}\n")

print("================================================================================================")


'''Question 2: Two-Sample t-Test
Perform a two-sample t-test to compare the means of two independent samples.

Generate two sample datasets each with 25 random values from normal distributions with means of 55 and 60, and a standard deviation of 8.
Perform an independent two-sample t-test to check if the means of the two samples are significantly different.
'''
# Question 2: Two-Sample t-Test
# Generate two sample datasets each with 25 random values from normal distributions with means of 55 and 60, and a standard deviation of 8.
np.random.seed(1)
sample1 = np.random.normal(55, 8, 25)
sample2 = np.random.normal(60, 8, 25)

# Perform an independent two-sample t-test to check if the means of the two samples are significantly different.
t_stat, p_value = stats.ttest_ind(sample1, sample2)
print(f"Two-Sample t-Test:\nT-statistic: {t_stat:.4f}, P-value: {p_value:.4f}\n")

print("================================================================================================")


'''Question 3: Chi-Squared Test
Objective: Perform a Chi-Squared test for independence.

Create a contingency table with observed frequencies for two categorical variables.

|-----------| Category A | Category B |
| Group 1 |     10   	    |     20   	 |
| Group 2 |     15    	    |     25     	 |

Perform a Chi-Squared test to determine if there is a significant association between the two categorical variables.

'''
# Question 3: Chi-Squared Test
observed = np.array([[10, 20], [15, 25]])

# Perform a Chi-Squared test to determine if there is a significant association between the two categorical variables.
chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"Chi-Squared Test:\nChi2 Statistic: {chi2_stat:.4f}, P-value: {p_value:.4f}, Degrees of Freedom: {dof}\nExpected Frequencies:\n{expected}\n")

print("================================================================================================")


'''Question 4: One-Way ANOVA
Objective: Perform a one-way ANOVA to compare means across multiple groups.

Generate three sample datasets each with 20 random values from normal distributions with means of 50, 55, and 60, and a standard deviation of 10.
Perform a one-way ANOVA to check if there are any significant differences in means across the three groups.
'''
# Question 4: One-Way ANOVA
# Generate three sample datasets each with 20 random values from normal distributions with means of 50, 55, and 60, and a standard deviation of 10.
np.random.seed(2)
group1 = np.random.normal(50, 10, 20)
group2 = np.random.normal(55, 10, 20)
group3 = np.random.normal(60, 10, 20)

# Perform a one-way ANOVA to check if there are any significant differences in means across the three groups.
f_stat, p_value = stats.f_oneway(group1, group2, group3)
print(f"One-Way ANOVA:\nF-statistic: {f_stat:.4f}, P-value: {p_value:.4f}\n")


print("================================================================================================")


'''
Question 5: Post-hoc Test using Tukey's HSD
Objective: Perform a post-hoc test using Tukey's HSD to identify which groups are significantly different.

Use the same datasets generated in the one-way ANOVA exercise.
Perform Tukey's HSD test to find out which pairs of group means are significantly different.
'''

# Question 5: Post-hoc Test using Tukey's HSD
# Combine the datasets into a single array and create labels for each group.
data = np.concatenate([group1, group2, group3])
labels = ['Group1'] * 20 + ['Group2'] * 20 + ['Group3'] * 20

# Perform Tukey's HSD test to find out which pairs of group means are significantly different.
tukey_result = pairwise_tukeyhsd(endog=data, groups=labels, alpha=0.05)
print("Tukey's HSD Test:\n", tukey_result)

print("================================================================================================")
