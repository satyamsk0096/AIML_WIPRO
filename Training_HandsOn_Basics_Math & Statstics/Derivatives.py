from scipy.misc import derivative

def f(X):
    return X**2 + 3**X + 2

X_Val = 1
f_prime = derivative(f, X_Val, dx=1e-6)

print("Function: f(X) = X**2 + 3**X + 2")
print(f"Derivative at X = {X_Val}: f'(X)=", f_prime)