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




