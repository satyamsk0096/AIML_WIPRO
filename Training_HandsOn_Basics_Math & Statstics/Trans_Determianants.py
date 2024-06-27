import numpy as np
matrix_A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
transpose_A = np.transpose(matrix_A)
print("Matrix A:\n", matrix_A)
print("Transpose of Matrix A:\n", transpose_A)
#==================================================================
matrix_A = np.array([
    [4, 7],
    [2, 6]
])
inverse_A = np.linalg.inv(matrix_A)
print("Matrix A:\n", matrix_A)
print("Inverse of Matrix A:\n", inverse_A)
#===================================================================
det_A = np.linalg.det(matrix_A)
print("Matrix A:\n", matrix_A)
print("Determinant of Matrix A:", det_A)
#========================================================================

eigenvalues, eigenvectors = np.linalg.eig(matrix_A)
print("Matrix A:\n", matrix_A)
print("Eigenvalues of Matrix A:\n", eigenvalues)
print("Eigenvectors of Matrix A:\n", eigenvectors)
#=========================================================================

