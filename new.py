import math

name = input("Enter your name: ")
print(f"hello , {name}!")


radius = float(input("Enter your radius: "))
area = math.pi*radius*radius
print(f"The area of the circle is: {area}")


length = float(input("Enter your length: "))
width = float(input("Enter your width: "))
area = length*width
perimeter = 2*(length+width)
print(f"The perimeter of the rectangle is: {perimeter}")
print(f"The area of the rectangle is: {area}")



num1 = int(input("Enter the first integer: "))
num2 = int(input("Enter the second integer: "))
num3 = int(input("Enter the third integer: "))
sum_of_numbers = num1 + num2 + num3
product_of_numbers = num1 * num2 * num3
average_of_numbers = sum_of_numbers / 3
print(f"Sum: {sum_of_numbers}")
print(f"Product: {product_of_numbers}")
print(f"Average: {average_of_numbers}")
