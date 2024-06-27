import numpy as np

vactor_A = np.array([1, 2])
vector_B = np.array([3,4])

dot_product = np.dot(vactor_A, vector_B)


#Maginitude of vectors
magnitude_a = np.linalg.norm(vactor_A)
magnitude_b = np.linalg.norm(vector_B)

cos_theta =  dot_product / (magnitude_a * magnitude_b)

print("Dot Product and its properties using Python")
print("Dot Product (A . B)", dot_product)
print("Maginitude of A:", magnitude_a)
print("Maginitude of B:", magnitude_b)
print("Cosine of the angle Between A and B:", cos_theta)