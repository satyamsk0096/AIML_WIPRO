def even_odd(n):
    even_sum = 0
    odd_sum = 0
    while n > 0:
        digit = n % 10
        if digit % 2 == 0:
            even_sum += digit
        else:
            odd_sum += digit
        n //= 10
    return even_sum - odd_sum

num = int(input("Enter the Number: "))
result = even_odd(num)
print("Difference between sums of even and odd digits:", result)
