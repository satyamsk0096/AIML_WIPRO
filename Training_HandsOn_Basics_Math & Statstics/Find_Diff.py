import numpy as np
from findiff import FinDiff


def g(x):
  """Defines a function g(x) = x^2."""
  return x**2


def h(x):
  """Defines a function h(x) = 3x + 1."""
  return 3*x + 1


def quotient_func(x):
  """Defines a function quotient_func(x) = g(x) / h(x)."""
  return g(x) / h(x)


def main():
  # Define the point for differentiation
  x_val = 1.0

  # Set a small step size for numerical differentiation
  dx = 1e-6

  # Create a FinDiff object for first-order differentiation
  d_dx = FinDiff(0, dx, 1)

  # Compute the derivative of the quotient function
  quotient_derivative = d_dx(quotient_func)(x_val)

  # Print the function and its derivative
  print("Quotient Function: g(x) / h(x) = (x**2) / (3*x + 1)")
  print(f"Quotient Derivative at x = {x_val}: (g / h)'(x) =", quotient_derivative)


if __name__ == "__main__":
  main()
