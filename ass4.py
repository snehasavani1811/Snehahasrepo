'''
num = 1
while num <= 1000:
    if num % 3 == 0:
        print(num)
    num += 1



INCH_TO_CM = 2.54
while True:
    inches = float(input("Enter a value in inches (negative value to quit): "))
    if inches < 0:
        print("Negative value entered. Exiting the program.")
        break
    centimeters = inches * INCH_TO_CM
    print(f"{inches} inches is equal to {centimeters} centimeters.\n")





numbers = []
while True:
    user_input = input("Enter a number (or press Enter to quit): ")
    if user_input == "":
        break
    try:
        number = float(user_input)
        numbers.append(number)
    except ValueError:
        print("Please enter a valid number.")
if numbers:
    smallest = min(numbers)
    largest = max(numbers)
    print(f"The smallest number is: {smallest}")
    print(f"The largest number is: {largest}")
else:
    print("No numbers were entered.")

import random
random_number = random.randint(1, 10)
while True:
    user_input = input("Guess the number (between 1 and 10): ")
    try:
        guess = int(user_input)
        if guess < random_number:
            print("Too low! Try again.")
        elif guess > random_number:
            print("Too high! Try again.")
        else:
            print("Correct! You guessed the number.")
            break
    except ValueError:
        print("Please enter a valid integer.")





correct_username = "Sneha"
correct_password = "Savani@sneha"
attempts = 0
max_attempts = 5
while attempts < max_attempts:
    username = input("Enter username: ")
    password = input("Enter password: ")
    if username == correct_username and password == correct_password:
        print("Welcome")
        break
    else:
        attempts += 1
        print(f"Incorrect username or password. Attempt {attempts}/{max_attempts}.")
if attempts == max_attempts:
    print("Access denied")





import random
def approximate_pi(num_points):
    points_inside_circle = 0
    for _ in range(num_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x ** 2 + y ** 2 < 1:
            points_inside_circle += 1
    pi_approximation = 4 * points_inside_circle / num_points
    return pi_approximation
num_points = int(input("Enter the number of random points to generate: "))
pi_value = approximate_pi(num_points)
print(f"Approximation of pi after {num_points} random points: {pi_value}")
'''







import random
def roll_dice():
    return random.randint(1, 6)
def main():
    result = 0
    while result != 6:
        result = roll_dice()
        print(f"Rolled: {result}")
if __name__ == "__main__":
    main()





import random
def roll_dice(sides):
    return random.randint(1, sides)
def main():
    sides = int(input("Enter the number of sides on the dice: "))

    result = 0
    while result != sides:
        result = roll_dice(sides)
        print(f"Rolled: {result}")
if __name__ == "__main__":
    main()





def gallons_to_liters(gallons):
    """Convert gallons to liters."""
    liters = gallons * 3.78541
    return liters
def main():
    while True:
        try:
            gallons = float(input("Enter the volume in gallons (negative to quit): "))
            if gallons < 0:
                print("Exiting the program.")
                break
            liters = gallons_to_liters(gallons)
            print(f"{gallons} gallons is equal to {liters:.2f} liters.")
        except ValueError:
            print("Please enter a valid number.")
if __name__ == "__main__":
    main()






def sum_of_list(numbers):
    """Return the sum of all integers in the list."""
    total = sum(numbers)
    return total
def main():
    integer_list = [1, 2, 3, 4, 5]
    total_sum = sum_of_list(integer_list)
    print(f"The sum of the list {integer_list} is: {total_sum}")
if __name__ == "__main__":
    main()




def remove_odd_numbers(numbers):
    """Return a new list containing only the even numbers from the original list."""
    even_numbers = [num for num in numbers if num % 2 == 0]  # List comprehension to filter even numbers
    return even_numbers
def main():
    original_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    filtered_list = remove_odd_numbers(original_list)
    print(f"Original list: {original_list}")
    print(f"List with odd numbers removed: {filtered_list}")
if __name__ == "__main__":
    main()




import math
def calculate_unit_price(diameter, price):
    """Calculate the unit price of the pizza per square meter."""
    radius = diameter / 2  # Radius in centimeters
    area = math.pi * (radius ** 2)  # Area in square centimeters
    area_in_square_meters = area / 10000  # Convert area to square meters
    unit_price = price / area_in_square_meters  # Unit price in euros per square meter
    return unit_price
def main():
    diameter1 = float(input("Enter the diameter of the first pizza (in cm): "))
    price1 = float(input("Enter the price of the first pizza (in euros): "))
    diameter2 = float(input("Enter the diameter of the second pizza (in cm): "))
    price2 = float(input("Enter the price of the second pizza (in euros): "))
    unit_price1 = calculate_unit_price(diameter1, price1)
    unit_price2 = calculate_unit_price(diameter2, price2)
    print(f"\nUnit price of the first pizza: {unit_price1:.2f} euros/m²")
    print(f"Unit price of the second pizza: {unit_price2:.2f} euros/m²")
    if unit_price1 < unit_price2:
        print("The first pizza provides better value for money.")
    elif unit_price1 > unit_price2:
        print("The second pizza provides better value for money.")
    else:
        print("Both pizzas provide  the same value for  money.")
if __name__ == "__main__":
    main()
