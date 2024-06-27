import numpy as np

# Sample Space and Events
sample_space = {'Heads', 'Tails'}
event = {'Heads'}

print(f"Sample Space: {sample_space}")
print(f"Event: {event}")
#=============
sample_space = ['H', 'T']
probabilities = [0.5, 0.5]

# Complement Rule
P_H = 0.5
P_not_H = 1 - P_H
print(f"P(H): {P_H}")
print(f"P(not H): {P_not_H}")

# Addition Rule
P_A = 0.4
P_B = 0.3
P_A_and_B = 0.1
P_A_or_B = P_A + P_B - P_A_and_B
print(f"P(A or B): {P_A_or_B}")

# Multiplication Rule
P_A_given_B = 0.5
P_B = 0.3
P_A_and_B = P_B * P_A_given_B
print(f"P(A and B): {P_A_and_B}")

# Conditional Probability
P_E_and_F = 0.1
P_F = 0.3
P_E_given_F = P_E_and_F / P_F
print(f"P(E|F): {P_E_given_F}")

# Bayes' Theorem
P_F_given_E = 0.5
P_E = 0.4
P_F = 0.3
P_E_given_F = (P_F_given_E * P_E) / P_F
print(f"P(E|F) using Bayes' Theorem: {P_E_given_F}")

# Independence Check
P_E = 0.4
P_F = 0.3
P_E_and_F = 0.12
are_independent = np.isclose(P_E_and_F, P_E * P_F)
print(f"Are E and F independent? {are_independent}")