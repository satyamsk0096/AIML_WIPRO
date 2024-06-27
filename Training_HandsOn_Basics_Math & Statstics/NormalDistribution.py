#PROP & VAL
from scipy.stats import norm

# Normal Distribution
mu, sigma = 0, 1
norm_rv = norm(mu, sigma)

# Mean
mean = norm_rv.mean()
print(f"Mean: {mean}")

# Variance
variance = norm_rv.var()
print(f"Variance: {variance}")

# Skewness
skewness = norm_rv.stats(moments='s')
print(f"Skewness: {skewness}")

# Kurtosis
kurtosis = norm_rv.stats(moments='k')
print(f"Kurtosis: {kurtosis}")
 