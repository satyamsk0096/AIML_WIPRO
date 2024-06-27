
#GOODNESS FIT
import numpy as np
from scipy import stats

# Observed frequencies
observed = np.array([8, 12, 11, 9, 10, 10])  #60 times
#observed = np.array([18, 5, 10, 12, 10, 5])  #60 times

# Expected frequencies (assuming a fair die)
expected = np.array([10, 10, 10, 10, 10, 10])  #60 times

# Perform the Chi-Squared Goodness of Fit test
chi_squared_stat, p_value = stats.chisquare(observed, f_exp=expected)

# Degrees of freedom
degrees_of_freedom = len(observed) - 1

# Significance level (alpha)
alpha = 0.05
critical_value = 10

print(f"Chi-Squared Statistic: {chi_squared_stat}")
print(f"P-value: {p_value}")
print(f"Degrees of Freedom: {degrees_of_freedom}")

# Decision
if p_value < alpha:
    print("Reject the null hypothesis: The observed frequencies do not match the expected frequencies.")
else:
    print("Fail to reject the null hypothesis: The observed frequencies match the expected frequencies.")

if chi_squared_stat > critical_value:
    print('Observed freq may not be a good fit')


#==============================================================

#TEST OF INDEPENDENCE
import numpy as np
from scipy import stats

# Create a contingency table
# Rows: Age Group (Under 30, 30-50, Over 50)
# Columns: Product Preference (Preferred, Not Preferred)
observed = np.array([[30, 10], [35, 15], [25, 20]])

# Perform the Chi-Squared Test of Independence
chi_squared_stat, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"Chi-Squared Statistic: {chi_squared_stat}")
print(f"P-value: {p_value}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies:\n{expected}")

# Significance level (alpha)
alpha = 0.05

# Decision
if p_value < alpha:
    print("Reject the null hypothesis: There is an association between age group and product preference.")
else:
    print("Fail to reject the null hypothesis: There is no association between age group and product preference.")

#=========================================================