import numpy as np

# Population data
population = np.random.normal(loc=50, scale=10, size=10000)

# Simple Random Sampling
sample_size = 100
sample = np.random.choice(population, sample_size)

# Sample mean and standard deviation
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)

print(f"Sample Mean: {sample_mean}")
print(f"Sample Standard Deviation: {sample_std}")

# Sampling Distribution of the Sample Mean
num_samples = 1000
sample_means = [np.mean(np.random.choice(population, sample_size))
                for _ in range(num_samples)]

# Mean and standard deviation of the sampling distribution
sampling_mean = np.mean(sample_means)
sampling_std = np.std(sample_means)

print(f"Sampling Distribution Mean: {sampling_mean}")
print(f"Sampling Distribution Standard Deviation: {sampling_std}")

#===========================================================

import scipy.stats as stats

# Sample data
data = np.random.normal(loc=50, scale=10, size=100)

# Point estimate (sample mean)
point_estimate = np.mean(data)

# Confidence interval
confidence_level = 0.95
degrees_freedom = len(data) - 1
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)
confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_std/np.sqrt(len(data)))

print(f"Point Estimate (Sample Mean): {point_estimate}")
print(f"{confidence_level*100}% Confidence Interval: {confidence_interval}")

#============================================================================

import scipy.stats as stats

# Sample data
data = np.random.normal(loc=50, scale=10, size=100)

# Hypothesis test
# Null hypothesis: mean = 50
# Alternative hypothesis: mean != 50
null_hypothesis_mean = 50
t_statistic, p_value = stats.ttest_1samp(data, null_hypothesis_mean)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")


#===============================================================


import numpy as np
import scipy.stats as stats

# Sample data
data = np.random.normal(loc=50, scale=10, size=100)

# Confidence interval for the mean
confidence_level = 0.95
degrees_freedom = len(data) - 1
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)
confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_std/np.sqrt(len(data)))

print(f"{confidence_level*100}% Confidence Interval: {confidence_interval}")

# Significance level and p-value
# Null hypothesis: mean = 50
null_hypothesis_mean = 50
t_statistic, p_value = stats.ttest_1samp(data, null_hypothesis_mean)

print(f"Significance Level (alpha): 0.05")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")

#==========================================================
