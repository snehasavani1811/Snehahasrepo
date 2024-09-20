import random
num_dice = int(input("How many dice would you like to roll? "))
total_sum = 0
for i in range(num_dice):
    roll = random.randint(1, 6)
    print(f"Roll {i+1}: {roll}")
    total_sum += roll
print(f"Total sum of rolls: {total_sum}")


numbers = []
while True:
    user_input = input("Enter a number (or press Enter to quit): ")
    if user_input == "":
        break

    try:
        number = int(user_input)
        numbers.append(number)
    except ValueError:
        print("Please enter a valid number.")

numbers.sort(reverse=True)
print(f"The five greatest numbers are: {numbers[:9]}")


def is_prime(n):

    if n < 2:
        return False

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
number = int(input("Enter an integer: "))

if is_prime(number):
    print(f"{number} is a prime number.")
else:
    print(f"{number} is not a prime number.")

cities = []
for i in range(5):
    city = input(f"Enter the name of city {i+1}: ")
    cities.append(city)
print("\nThe cities you entered are:")
for city in cities:
    print(city)