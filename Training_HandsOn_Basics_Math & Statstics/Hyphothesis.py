'''
1-SAMPLE T-TEST

Examine whether a new teaching method affects the average test scores of students. 
We hypothesize that the new teaching method has not changed the average test score
from the traditional method, which is known to be 75.
 
One-Sample t-Test for Average Test Scores
Formulating Hypotheses
•  Null Hypothesis (H0): The mean test score with the new teaching method is equal to 75.
H0 : μ <= 75
•  Alternative Hypothesis (H1): The mean test score with the new teaching method is different from 75.
H1 : μ > 75

'''

from scipy import stats

# Sample data: test scores of students taught using the new method
test_scores = [78, 74, 82, 70, 75, 77, 79, 81, 73, 76]

population_mean = 75
alpha = 0.05

t_statistic, p_value = stats.ttest_1samp(test_scores, population_mean)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

if p_value < alpha:
    print("Reject the null hypothesis: The new teaching method has changed the average test score.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in average test scores.")

#=====================================================================

'''
2-SAMPLE T-TEST ==> Independent 
H0 : There is no significant change between the 2 methods  
There is no significant difference between the two teaching methods
H1 : There is significant change between the 2 methods and method 2 is better
There is a significant difference between the two teaching methods.
'''

from scipy import stats

# Sample data: test scores from two different teaching methods
method_1_scores = [78, 74, 82, 70, 75, 77, 79, 81, 73, 76]
method_2_scores = [80, 85, 78, 82, 84, 79, 81, 83, 80, 82]

# Perform independent samples t-test
t_statistic, p_value = stats.ttest_ind(method_1_scores, method_2_scores)

alpha = 0.05

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the two teaching methods.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the two teaching methods.")


#=======================================================================================



from scipy import stats


'''
2-SAMPLE T-TEST ==> Dependent / Paired

H0: There is no significant difference between the test scores
 before and after the new teaching method.
H1: There is a significant difference between the test scores 
before and after the new teaching method.
'''
# Sample data: test scores before and after using a new teaching method
before_scores = [78, 74, 82, 70, 75, 77, 79, 81, 73, 76]
after_scores = [80, 85, 78, 82, 84, 79, 81, 83, 80, 82]

# Perform paired samples t-test
t_statistic, p_value = stats.ttest_rel(before_scores, after_scores)

alpha = 0.05

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the test scores before and after the new teaching method.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the test scores before and after the new teaching method.")

#===================================================
