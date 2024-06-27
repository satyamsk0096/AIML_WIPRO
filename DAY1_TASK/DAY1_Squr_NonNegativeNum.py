import math

def calculate_square_root():
    try:
        num = float(input("Enter a non-negative number: "))
        if num < 0:
            raise ValueError("You must enter a non-negative number")
        result = math.sqrt(num)
    except ValueError as ve:
        print(f"Error: {ve}")
    except NameError:
        print("Error: Invalid input. Please enter a valid number.")
    else:
        print(f"The square root of {num} is: {result:.2f}")
    finally:
        print("Program execution completed.")

calculate_square_root()
