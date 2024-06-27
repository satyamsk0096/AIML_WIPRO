'''Exercise 1: Create different types of NumPy arrays and perform basic manipulations.

a. Create a 1-dimensional array:
Create a 1-dimensional array of integers from 0 to 9.
Print the array and its shape.

b. Create a 2-dimensional array:
Create a 2-dimensional array (3x3) with values from 1 to 9.
Print the array, its shape, and the sum of all elements.

c. Reshape the array:
Reshape the 1-dimensional array from step 1 into a 2x5 array.
Print the reshaped array and its shape.'''
import numpy as np
import time
import numba

# Exercise 1: Create different types of NumPy arrays and perform basic manipulations

# a. Create a 1-dimensional array
array_1d = np.arange(10)
print("1-dimensional array:", array_1d)
print("Shape:", array_1d.shape)

# b. Create a 2-dimensional array
array_2d = np.arange(1, 10).reshape(3, 3)
print("\n2-dimensional array:\n", array_2d)
print("Shape:", array_2d.shape)
print("Sum of all elements:", np.sum(array_2d))

# c. Reshape the array
reshaped_array = array_1d.reshape(2, 5)
print("\nReshaped array:\n", reshaped_array)
print("Shape:", reshaped_array.shape)



print("================================================================================================")





'''Exercise 2: Perform Basic and Advanced Array Operations

a. Array arithmetic:
Create two 1-dimensional arrays of integers from 1 to 5 and 6 to 10.
Perform element-wise addition, subtraction, multiplication, and division and Print the results.

b. Indexing and slicing:
Create a 5x5 array with values from 1 to 25.
Extract the subarray consisting of the first two rows and columns.
Print the extracted subarray.

c. Boolean indexing:
Create a 1-dimensional array of integers from 10 to 19.
Extract elements greater than 15.
Print the resulting array.'''

# Exercise 2: Perform Basic and Advanced Array Operations

# a. Array arithmetic
array1 = np.arange(1, 6)
array2 = np.arange(6, 11)

addition = array1 + array2
subtraction = array1 - array2
multiplication = array1 * array2
division = array1 / array2

print("\nAddition:", addition)
print("Subtraction:", subtraction)
print("Multiplication:", multiplication)
print("Division:", division)

# b. Indexing and slicing
array_5x5 = np.arange(1, 26).reshape(5, 5)
print("\nOriginal 5x5 array:\n", array_5x5)

subarray = array_5x5[:2, :2]
print("Extracted subarray:\n", subarray)

# c. Boolean indexing
array_1d_bi = np.arange(10, 20)
print("\nOriginal array:", array_1d_bi)

extracted_elements = array_1d_bi[array_1d_bi > 15]
print("Elements greater than 15:", extracted_elements)

print("================================================================================================")


'''Exercise 3: Use NumPy for Mathematical and Statistical Calculations

a. Mathematical functions:
Create an array of 10 evenly spaced values between 0 and 2π.
Compute the sine, cosine, and tangent of each value.
Print the results.

b. Statistical functions:
Create a 3x3 array with random integers between 1 and 100.
Compute the mean, median, standard deviation, and variance.
Print the results.'''
# Exercise 3: Use NumPy for Mathematical and Statistical Calculations

# a. Mathematical functions
values = np.linspace(0, 2 * np.pi, 10)
sine_values = np.sin(values)
cosine_values = np.cos(values)
tangent_values = np.tan(values)

print("\nValues:", values)
print("Sine:", sine_values)
print("Cosine:", cosine_values)
print("Tangent:", tangent_values)

# b. Statistical functions
array_random = np.random.randint(1, 101, size=(3, 3))
print("\nRandom 3x3 array:\n", array_random)

mean_value = np.mean(array_random)
median_value = np.median(array_random)
std_deviation = np.std(array_random)
variance = np.var(array_random)

print("Mean:", mean_value)
print("Median:", median_value)
print("Standard Deviation:", std_deviation)
print("Variance:", variance)

print("================================================================================================")


'''
Exercise 4: Implement Broadcasting and Vectorized Operations

a. Broadcasting:
Create a 3x1 array with values from 1 to 3.
Create a 1x3 array with values from 4 to 6.
Add the two arrays using broadcasting.
Print the resulting array.

b. Vectorized operations:
Create two large arrays of size 1,000,000 with random values.
Compute the element-wise product of the two arrays.
Print the time taken for the computation using vectorized operations.
'''
# Exercise 4: Implement Broadcasting and Vectorized Operations

# a. Broadcasting
array_3x1 = np.array([[1], [2], [3]])
array_1x3 = np.array([4, 5, 6])

broadcasted_sum = array_3x1 + array_1x3
print("\nResult of broadcasting addition:\n", broadcasted_sum)

# b. Vectorized operations
large_array1 = np.random.rand(1000000)
large_array2 = np.random.rand(1000000)

start_time = time.time()
elementwise_product = large_array1 * large_array2
end_time = time.time()

print("Time taken for vectorized operation:", end_time - start_time, "seconds")

print("================================================================================================")


'''Exercise 5: Optimize Performance Using Vectorization and Numba

a. Vectorization:
Create a function to compute the element-wise square of an array using a for loop.
Create another function to perform the same computation using NumPy vectorization.
Compare the performance of the two functions using a large array of size 1,000,000.

b. Numba:
Use the @numba.jit decorator to optimize the function from step 1 that uses a for loop.
Compare the performance of the Numba-optimized function with the vectorized NumPy function.
'''
# Exercise 5: Optimize Performance Using Vectorization and Numba

# a. Vectorization
def square_using_loop(arr):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] ** 2
    return result

def square_using_vectorization(arr):
    return arr ** 2

large_array = np.random.rand(1000000)

start_time = time.time()
square_using_loop(large_array)
end_time = time.time()
print("Time taken using loop:", end_time - start_time, "seconds")

start_time = time.time()
square_using_vectorization(large_array)
end_time = time.time()
print("Time taken using vectorization:", end_time - start_time, "seconds")

# b. Numba
@numba.jit
def square_using_numba(arr):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] ** 2
    return result

start_time = time.time()
square_using_numba(large_array)
end_time = time.time()
print("Time taken using Numba:", end_time - start_time, "seconds")

start_time = time.time()
square_using_vectorization(large_array)
end_time = time.time()
print("Time taken using vectorization:", end_time - start_time, "seconds")

print("================================================================================================")
