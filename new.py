class Car:
    def __init__(self, registration_number, max_speed):
        self.registration_number = registration_number
        self.max_speed = max_speed
        self.current_speed = 0
        self.travelled_distance = 0

def main():
    my_car = Car("ABC-123", 142)
    print(f"Registration Number: {my_car.registration_number}")
    print(f"Maximum Speed: {my_car.max_speed} km/h")
    print(f"Current Speed: {my_car.current_speed} km/h")
    print(f"Travelled Distance: {my_car.travelled_distance} km")
if __name__ == "__main__":
    main()








class Car:
    def __init__(self, registration_number, max_speed):
        self.registration_number = registration_number
        self.max_speed = max_speed
        self.current_speed = 0
        self.travelled_distance = 0

    def accelerate(self, speed_change):
        new_speed = self.current_speed + speed_change
        if new_speed > self.max_speed:
            self.current_speed = self.max_speed
        elif new_speed < 0:
            self.current_speed = 0
        else:
            self.current_speed = new_speed
def main():
    my_car = Car("ABC-123", 142)
    my_car.accelerate(30)
    print(f"Current Speed after +30 km/h: {my_car.current_speed} km/h")
    my_car.accelerate(70)
    print(f"Current Speed after +70 km/h: {my_car.current_speed} km/h")
    my_car.accelerate(50)
    print(f"Current Speed after +50 km/h: {my_car.current_speed} km/h")
    my_car.accelerate(-200)
    print(f"Current Speed after emergency brake (-200 km/h): {my_car.current_speed} km/h")
if __name__ == "__main__":
    main()





class Car:
    def __init__(self, registration_number, max_speed):
        self.registration_number = registration_number
        self.max_speed = max_speed
        self.current_speed = 0
        self.travelled_distance = 0
    def accelerate(self, speed_change):
        new_speed = self.current_speed + speed_change
        if new_speed > self.max_speed:
            self.current_speed = self.max_speed
        elif new_speed < 0:
            self.current_speed = 0
        else:
            self.current_speed = new_speed

    def drive(self, hours):
        distance_travelled = self.current_speed * hours
        self.travelled_distance += distance_travelled
def main():
    my_car = Car("ABC-123", 142)
    my_car.accelerate(30)
    print(f"Current Speed after +30 km/h: {my_car.current_speed} km/h")
    my_car.accelerate(70)
    print(f"Current Speed after +70 km/h: {my_car.current_speed} km/h")
    my_car.accelerate(50)
    print(f"Current Speed after +50 km/h: {my_car.current_speed} km/h")
    my_car.drive(1.5)
    print(f"Travelled Distance after 1.5 hours of driving: {my_car.travelled_distance} km")
    my_car.accelerate(-200)
    print(f"Current Speed after emergency brake (-200 km/h): {my_car.current_speed} km/h")
if __name__ == "__main__":
    main()
class InsufficientBalanceError(Exception):
    pass


class NegativeAmountError(Exception):
    pass
def main():
    try:
        balance_input = input("Enter your account balance: ")
        balance = float(balance_input)
        withdrawal_input = input("Enter the amount to withdraw: ")
        withdrawal_amount = float(withdrawal_input)
        if withdrawal_amount < 0:
            raise NegativeAmountError("Withdrawal amount cannot be negative.")
        if withdrawal_amount > balance:
            raise InsufficientBalanceError("Withdrawal amount exceeds account balance.")
        balance -= withdrawal_amount
        print(f"Withdrawal successful! New balance: {balance:.2f}")
    except ValueError:
        print("Invalid input! Please enter numeric values.")
    except NegativeAmountError as e:
        print(f"Error: {e}")
    except InsufficientBalanceError as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    main()


def write_notes(filename):
    # Write new notes to a file (overwrites existing content)
    with open(filename, 'w') as file:
        note = input("Enter your new note: ")
        file.write(note + '\n')
    print("Note written successfully.")





def read_notes(filename):
    try:
        with open(filename, 'r') as file:
            notes = file.read()
            if notes:
                print("\nExisting notes:")
                print(notes)
            else:
                print("No notes found.")
    except FileNotFoundError:
        print("No notes found. The file does not exist.")
def append_notes(filename):
    with open(filename, 'a') as file:
        note = input("Enter your additional note: ")
        file.write(note + '\n')
    print("Note appended successfully.")
def main():
    filename = "notes.txt"  # File to store notes
    while True:
        print("\nChoose an option:")
        print("1. Write new notes (overwrite existing)")
        print("2. Read existing notes")
        print("3. Append additional notes")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")
        if choice == '1':
            write_notes(filename)
        elif choice == '2':
            read_notes(filename)
        elif choice == '3':
            append_notes(filename)
        elif choice == '4':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")
if __name__ == "__main__":
    main()












