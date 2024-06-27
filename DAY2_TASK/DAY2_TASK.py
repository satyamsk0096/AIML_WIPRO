'''Question 1: Matrix Operations with NumPy

Create two 3x3 matrices A and B with random integer values between 1 and 10.
Compute the following:
The sum of A and B.
The difference between A and B.
The element-wise product of A and B.
The matrix product of A and B.
The transpose of matrix A.
The determinant of matrix A.
'''

import numpy as np
from scipy.linalg import solve
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.stats import skew, kurtosis, ttest_1samp

# Question 1
np.random.seed(0)
A = np.random.randint(1, 11, size=(3, 3))
B = np.random.randint(1, 11, size=(3, 3))
sum_AB = A + B
diff_AB = A - B
elem_product_AB = A * B
matrix_product_AB = np.dot(A, B)
transpose_A = A.T
det_A = np.linalg.det(A)

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Sum of A and B:\n", sum_AB)
print("Difference between A and B:\n", diff_AB)
print("Element-wise product of A and B:\n", elem_product_AB)
print("Matrix product of A and B:\n", matrix_product_AB)
print("Transpose of matrix A:\n", transpose_A)
print("Determinant of matrix A:\n", det_A)


print("================================================================================================")



'''Question 2: Solving Linear Equations with SciPy

Given the system of equations:
2x + 3y = 8
3x + 4y = 11

Represent the system of equations in matrix form AX = B.
Use scipy.linalg.solve to find the values of x and y.
'''

# Question 2
A_eq = np.array([[2, 3], [3, 4]])
B_eq = np.array([8, 11])
solution = solve(A_eq, B_eq)
x, y = solution
print(f"Solution for the system of equations: x = {x}, y = {y}")

print("================================================================================================")

'''Question 3: Calculus with SciPy

Define the function f(x) = x^3 + 2x^2 + x + 1.
Compute the first and second derivatives of f(x) at x = 1.
Compute the definite integral of f(x) from x = 0 to x = 2.
'''
# Question 3
def f(x):
    return x**3 + 2*x**2 + x + 1

first_derivative = derivative(f, 1, dx=1e-6)
second_derivative = derivative(f, 1, dx=1e-6, n=2)
integral, _ = quad(f, 0, 2)

print("First derivative of f(x) at x=1:", first_derivative)
print("Second derivative of f(x) at x=1:", second_derivative)
print("Definite integral of f(x) from 0 to 2:", integral)


print("================================================================================================")

'''Question 4: Descriptive Statistics with NumPy and SciPy

Create a dataset with 20 random values between 1 and 100.
Compute the following statistics for the dataset:
Mean
Median
Standard deviation
Variance
Skewness
Kurtosis
'''

# Question 4
dataset = np.random.randint(1, 101, size=20)

print("Dataset:", dataset)
mean = np.mean(dataset)
median = np.median(dataset)
std_dev = np.std(dataset)
variance = np.var(dataset)
skewness = skew(dataset)
kurt = kurtosis(dataset)

print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)
print("Variance:", variance)
print("Skewness:", skewness)
print("Kurtosis:", kurt)

print("================================================================================================")

'''Question 5: Hypothesis Testing with SciPy
Generate a sample dataset of 30 random values from a normal distribution with mean 50 and standard deviation 5.
Perform a one-sample t-test to check if the sample mean is significantly different from 50.'''

# Question 5
sample = np.random.normal(50, 5, 30)
t_stat, p_value = ttest_1samp(sample, 50)

print("Sample dataset:", sample)
print("T-statistic:", t_stat)
print("P-value:", p_value)


print("================================================================================================")
