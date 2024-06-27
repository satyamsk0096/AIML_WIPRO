import numpy as np
'''
Random Arrays
rand: Creates an array of given shape with random values between 0 and 1.
randn: Creates an array of given shape with random values from a 
                standard normal distribution.
randint: Creates an array with random integers within a specified range.

'''

array_rand = np.random.rand(2, 3)
print("Random array with rand:\n", array_rand)

array_randn = np.random.randn(2, 3)
print("Random array with randn:\n", array_randn)

array_randint = np.random.randint(0, 10, (2, 3))
print("Random array with randint:\n", array_randint)


'''
Indexing and Slicing
Reshaping Arrays
Change the shape of an array without changing its data using the
 reshape() method.

'''

array = np.array([[1, 2, 3], [4, 5, 6]])

# Indexing
print("Element at [0, 1]:", array[0, 1])

# Slicing
print("First row:", array[0, :])
print("First column:", array[:, 0])
print("Sub-array:", array[0:2, 1:3])

array = np.arange(6)
print("Original array:", array)

reshaped_array = array.reshape((2, 3))
print("Reshaped array:\n", reshaped_array)


#===================================================================