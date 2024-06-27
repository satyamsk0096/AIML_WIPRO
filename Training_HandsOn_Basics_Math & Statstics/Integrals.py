import sympy as sp

# Define the variable and the function
x = sp.symbols('x')
f = x**2 + 3*x + 2

# Compute the indefinite integral
F = sp.integrate(f, x)

print("Indefinite integral of f(x) = x**2 + 3*x + 2:")
sp.pprint(F)

#=============================================================

import sympy as sp

# Define the variable and the function
x = sp.symbols('x')
f = x**2 + 3*x + 2

# Compute the definite integral from 1 to 3
a, b = 1, 3
def_integral = sp.integrate(f, (x, a, b))
print(f"Definite integral of f(x) = x**2 + 3*x + 2 from {a} to {b}:")
sp.pprint(def_integral)

#==========================================================================

import sympy as sp

# Define the variables and the functions
x = sp.symbols('x')
f = x**2
g = 3*x
h = x**2 + 3*x

# Power rule
int_power = sp.integrate(x**2, x)
print("Integral of x^2:")
sp.pprint(int_power)

# Constant multiple rule
int_const_multiple = sp.integrate(3*x, x)
print("Integral of 3x:")
sp.pprint(int_const_multiple)

# Sum rule
int_sum = sp.integrate(h, x)
print("Integral of (x^2 + 3x):")
sp.pprint(int_sum)


#+========================================================================