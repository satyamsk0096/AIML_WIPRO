import sympy as sp

x, y = sp.symbols('x y')
f = x**2 + y**2

# Compute the partial derivatives
f_x = sp.diff(f, x)
f_y = sp.diff(f, y)

print(f"Partial derivative with respect to x: {f_x}")
print(f"Partial derivative with respect to y: {f_y}")

# Compute the gradient
gradient_f = (f_x, f_y)
print(f"Gradient of f: {gradient_f}")

#=======================================================================

# Define the variables and the vector-valued function
x, y = sp.symbols('X Y')
F = sp.Matrix([x**2 + y**2, sp.sin(x) + sp.cos(y)])
print('F: ', F)
print(x,y)

# Compute the Jacobian matrix
J = F.jacobian([x, y])
print("Jacobian matrix:")
sp.pprint(J)


x, y = sp.symbols('x y')
f = x**2 + y**2
print(x,y)
print('f: ',f)
# Compute the Hessian matrix
H = sp.hessian(f, (x, y))
print("Hessian matrix:")
sp.pprint(H)

#=====================================================================


 