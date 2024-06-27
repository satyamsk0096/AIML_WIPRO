# Definition and notation of matrices using Python
import numpy as np

matrix_A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("Matrix A:\n", matrix_A)


#======================================================

# Matrix addition and subtraction using Python
import numpy as np

matrix_A = np.array([
    [1, 2],
    [3, 4]
])

matrix_B = np.array([
    [5, 6],
    [7, 8]
])

matrix_add = matrix_A + matrix_B
matrix_sub = matrix_A - matrix_B

print("Matrix A:\n", matrix_A)
print("Matrix B:\n", matrix_B)
print("Matrix Addition (A + B):\n", matrix_add)
print("Matrix Subtraction (A - B):\n", matrix_sub)

#=====================================================

# Scalar multiplication using Python
import numpy as np

matrix = np.array([
    [1, 2],
    [3, 4]
])

scalar = 3

matrix_scaled = scalar * matrix

print("Original Matrix:\n", matrix)
print("Scalar:", scalar)
print("Scaled Matrix:\n", matrix_scaled)


#====================================================

# Matrix multiplication and its properties using Python
import numpy as np

matrix_A = np.array([
    [1, 2],
    [3, 4]
])

matrix_B = np.array([
    [2, 0],
    [1, 3]
])

# Matrix multiplication
matrix_mul = np.dot(matrix_A, matrix_B)

# Identity matrix of size 2x2
identity_matrix = np.eye(2)

# Properties: A * I = A and I * A = A
left_identity = np.dot(identity_matrix, matrix_A)
right_identity = np.dot(matrix_A, identity_matrix)

print("Matrix A:\n", matrix_A)
print("Matrix B:\n", matrix_B)
print("Matrix Multiplication (A * B):\n", matrix_mul)
print("Identity Matrix:\n", identity_matrix)
print("A * I:\n", left_identity)
print("I * A:\n", right_identity)

# Verify non-commutativity (A * B != B * A)
matrix_mul_BA = np.dot(matrix_B, matrix_A)
print("Matrix Multiplication (B * A):\n", matrix_mul_BA)

#=====================================================================
