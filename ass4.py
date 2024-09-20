import random
num_dice = int(input("How many dice would you like to roll? "))
total_sum = 0
for i in range(num_dice):
    roll = random.randint(1, 6)
    print(f"Roll {i+1}: {roll}")
    total_sum += roll
print(f"Total sum of rolls rolls: {total_sum}")


