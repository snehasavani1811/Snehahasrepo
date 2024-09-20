'''def greet():
    print("Hello!")
    return
print("A new day starts with a greeting.")
greet()
print("Now we can  move to other business.")'''
from importlib import simple

'''def greet(times):
    for i in range(times):
        print("Round"+ str(i+1)+"of saying hello.")
    return

print(" A new day starts with greeting.")
greet(5)
print("Let's greet some more.")
greet(2)

def change():
    city="Vantaa"
    print("At the end of the function:"+city)
    return
city="Helsinki"
print("At the beginning in the main program:"+city)
change()
print("At the end of the main program:"+city)

def sum_of_squares(first, second):
    result=first**2+second**2
    return result
number1 = float(input("Enter the first number: "))
number2 = float(input("Enter the second number: "))
result=sum_of_squares(number1, number2)
print(f"The sum of squares of numbers {number1:.1f} and {number2:.3f} is {result:.3f}")

def inventory(items):
    print("You have the following items:")
    for item in items:
        print("-" + item)
    items.clear()
    return
backpack=["Water bottle", "Map","Compass"]
inventory(backpack)
backpack.append("Swiss Army Knife")
inventory(backpack)

user=input("Enter your name: ")
while username != "Stop"
    print("username")
    username=input("Enter your name: ")

number = 1
while number<5:
    print(number)
print("All Ready.")

import random
dice1 = dice2 = rolls = 0
while (dice1!=6 or dice2!=6):
    dice1 = random.randint(1,6)
    dice2 = random.randint(1,6)
    rolls = rolls + 1
print(f"Rolled{rolls:d} times.")

import random
rounds = 0
total_rolls = 0
while rounds < 10000:
    dice1 = dice2 = rolls = 0
    while (dice1 != 6 or dice2 != 6):
        dice1 = random.randint(1, 6)
        dice2 = random.randint(1, 6)
        rolls = rolls + 1
    print(f"Rolled{rolls:d} times.")
    rounds = rounds + 1
    total_rolls = total_rolls + rolls
average_rolls = total_rolls/rounds
print(f"Average rolls required: {average_rolls:.2f}")

first = 2
while first <= 5:
    second = 4
    while second <= 5:
        print(f"{first} times{second} is {first*second}")
        second = second + 1
        first = first + 1


command = input("Enter command: ")
while command != "exit":
    if command == "MAYDAY":
        break
    print("Executing command:" + command)
    command = input("Enter command: ")
else:
    print("Goodbye.")
print("Execution stopped.")'''


def calculate_simple_interest(principal, interest, rate):
    simple_interest = principal * interest * rate
    return simple interest
def calculate_compound_interest(principal, interest, rate):
    amount = principal * (1 + rate/100) ** rate
    compound_interest = amount - principal
    return compound_interest
principal = input("Enter principal: ")
interest = input("Enter interest: ")
rate = input("Enter rate: ")
time = input("Enter time: ")
print(f"Simple Interest: {simple_interest}")
print(f"Compound Interest: {compund_interest}")
