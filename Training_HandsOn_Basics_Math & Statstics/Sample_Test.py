from scipy import stats

'''
1-SAMPLE T-TEST

Formulating Hypotheses:
•  H0 : μ = 300
•  H1 : μ ≠ 300

The t-test checks whether the sample mean is significantly 
different from the population mean. 
Since the sample mean is very close to 300, the p-value will be higher,
indicating insufficient evidence to reject the null hypothesis 
at the 0.05 significance level.
'''

# lifespans of new batteries (in hours)
#data = [310, 320, 290, 330, 340, 300, 310, 320, 305, 325]

data = [295, 302, 298, 301, 299, 300, 297, 303, 296, 300]

# Population mean of old batteries
population_mean = 300
# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(data, population_mean)
alpha = 0.05
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")
if p_value < alpha:
    print("Reject the null hypothesis \n The new battery lasts longer than the old battery.")
else:
    print("Fail to reject the null hypothesis \n There is no significant difference in battery life.")