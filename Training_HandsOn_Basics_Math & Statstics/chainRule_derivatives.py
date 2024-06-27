import numpy as np

from scipy.misc import derivative


def g(x):
    return np.sin(x)

def f(u):
    return np.exp(u)

def composite_func(x):
    return f(g(x))

x_val =1 
inner_derivative = derivative(g, x_val, dx=1)
outer_derivative = derivative(f, g(x_val), dx=1)
chain_derivative = outer_derivative * inner_derivative

print("Composite Function: f(g(x)) = exp(sin(x))")
print(f"Chain Rule Derivative at x = {x_val}: dy/dx =", chain_derivative)


