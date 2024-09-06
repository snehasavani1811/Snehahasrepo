name = input("Enter your name: ")
print(f"hello , {name}!")



import math
radius = float(input("Enter the radius of the circle: "))
area = math.pi * radius ** 2
print(f"The area of the circle with radius {radius} is {area:.2f}")



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



LOT_TO_GRAMS = 13.3
POUND_TO_LOTS = 32
TALENT_TO_POUNDS = 20
talents = int(input("Enter the number of talents: "))
pounds = int(input("Enter the number of pounds: "))
lots = int(input("Enter the number of lots: "))
total_lots = (talents * TALENT_TO_POUNDS * POUND_TO_LOTS) + (pounds * POUND_TO_LOTS) + lots
total_grams = total_lots * LOT_TO_GRAMS
kilograms = int(total_grams // 1000)
grams = total_grams % 1000
print(f"The total weight is {kilograms} kilograms and {grams:.2f} grams.")


import random
three_digit_code = [random.randint(0, 9) for _ in range(3)]
four_digit_code = [random.randint(1, 6) for _ in range(4)]
print(f"3-digit code: {''.join(map(str, three_digit_code))}")
print(f"4-digit code: {''.join(map(str, four_digit_code))}")



SIZE_LIMIT = 42
length = float(input("Enter the length of the zander (in cm): "))
if length < SIZE_LIMIT:
    below_limit = SIZE_LIMIT - length
    print(f"The zander is {below_limit:.1f} cm below the size limit. Please release it back into the lake.")
else:
    print("The zander meets the size limit. You can keep it.")





cabin_class = input("Enter the cabin class (LUX, A, B, C): ").upper()
if cabin_class == "LUX":
    print("LUX: upper-deck cabin with a balcony.")
elif cabin_class == "A":
    print("A: above the car deck, equipped with a window.")
elif cabin_class == "B":
    print("B: windowless cabin above the car deck.")
elif cabin_class == "C":
    print("C: windowless cabin below the car deck.")
else:
    print("Invalid cabin class.")




gender = input("Enter your biological gender (male/female): ").lower()
hemoglobin = float(input("Enter your hemoglobin value (g/l): "))
FEMALE_MIN = 117
FEMALE_MAX = 155
MALE_MIN = 134
MALE_MAX = 167
if gender == "female":
    if hemoglobin < FEMALE_MIN:
        print("Your hemoglobin value is low.")
    elif FEMALE_MIN <= hemoglobin <= FEMALE_MAX:
        print("Your hemoglobin value is normal.")
    else:
        print("Your hemoglobin value is high.")
elif gender == "male":
    if hemoglobin < MALE_MIN:
        print("Your hemoglobin value is low.")
    elif MALE_MIN <= hemoglobin <= MALE_MAX:
        print("Your hemoglobin value is normal.")
    else:
        print("Your hemoglobin value is high.")
else:
    print("Invalid gender input. Please enter 'male' or 'female'.")



year = int(input("Enter a year: "))
if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print(f"{year} is a leap year.")
else:
    print(f"{year} is not a leap year.")








