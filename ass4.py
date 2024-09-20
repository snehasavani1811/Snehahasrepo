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





'''import random
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
    print(city)'''



seasons = ("Winter", "Spring", "Summer", "Autumn")
def get_season(month):
    if month in (12, 1, 2):
        return seasons[0]
    elif month in (3, 4, 5):
        return seasons[1]
    elif month in (6, 7, 8):
        return seasons[2]
    elif month in (9, 10, 11):
        return seasons[3]
    else:
        return None
month = int(input("Enter the month number (1-12): "))
season = get_season(month)
if season:
    print(f"The month {month} corresponds to {season}.")
else:
    print("Invalid month number. Please enter a number between 1 and 12.")



names_set = set()
while True:
    name = input("Enter a name (or press Enter to finish): ")
    if name == "":
        break
    if name in names_set:
        print("Existing name")
    else:
        print("New name")
        names_set.add(name)
print("\nList of names entered:")
for name in names_set:
    print(name)



airports = {}
def add_airport():
    """Adds a new airport to the dictionary."""
    icao = input("Enter the ICAO code of the airport: ").upper()
    name = input("Enter the name of the airport: ")
    if icao in airports:
        print("This airport is already in the system.")
    else:
        airports[icao] = name
        print(f"Airport {name} added with ICAO code {icao}.")
def fetch_airport():
    """Fetches the airport name using the ICAO code."""
    icao = input("Enter the ICAO code of the airport: ").upper()
    if icao in airports:
        print(f"The name of the airport with ICAO code {icao} is {airports[icao]}.")
    else:
        print("Airport not found.")
def main():
    """Main program loop to interact with the user."""
    while True:
        print("\nOptions:")
        print("1. Enter a new airport")
        print("2. Fetch an airport by ICAO code")
        print("3. Quit")
        choice = input("Choose an option (1/2/3): ")
        if choice == "1":
            add_airport()
        elif choice == "2":
            fetch_airport()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please select 1, 2, or 3.")
if __name__ == "__main__":
    main()





seasons = ("Winter", "Spring", "Summer", "Autumn")
def get_season(month):
    if month in (12, 1, 2):
        return seasons[0]
    elif month in (3, 4, 5):
        return seasons[1]
    elif month in (6, 7, 8):
        return seasons[2]
    elif month in (9, 10, 11):
        return seasons[3]
    else:
        return None
month = int(input("Enter the month number (1-12): "))
season = get_season(month)
if season:
    print(f"The month {month} corresponds to {season}.")
else:
    print("Invalid month number. Please enter a number between 1 and 12.")



names_set = set()
while True:
    name = input("Enter a name (or press Enter to finish): ")
    if name == "":
        break
    if name in names_set:
        print("Existing name")
    else:
        print("New name")
        names_set.add(name)
print("\nList of names entered:")
for name in names_set:
    print(name)



airports = {}
def add_airport():
    """Adds a new airport to the dictionary."""
    icao = input("Enter the ICAO code of the airport: ").upper()
    name = input("Enter the name of the airport: ")
    if icao in airports:
        print("This airport is already in the system.")
    else:
        airports[icao] = name
        print(f"Airport {name} added with ICAO code {icao}.")
def fetch_airport():
    """Fetches the airport name using the ICAO code."""
    icao = input("Enter the ICAO code of the airport: ").upper()
    if icao in airports:
        print(f"The name of the airport with ICAO code {icao} is {airports[icao]}.")
    else:
        print("Airport not found.")
def main():
    """Main program loop to interact with the user."""
    while True:
        print("\nOptions:")
        print("1. Enter a new airport")
        print("2. Fetch an airport by ICAO code")
        print("3. Quit")
        choice = input("Choose an option (1/2/3): ")
        if choice == "1":
            add_airport()
        elif choice == "2":
            fetch_airport()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please select 1, 2, or 3.")
if __name__ == "__main__":
    main()

