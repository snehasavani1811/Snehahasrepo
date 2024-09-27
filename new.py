
'''file = open("example.txt","w")
file.write("Everything happens for a reason")
file.close()

file = open("example.txt","r")
content=file.read()
print(content)
file.close()


file = open("sneha.txt", "r")
content = file.read()
print(content)
file.close()


with open("sneha.txt", "r") as file:
    content=file.read()
    print(content)


try:
    num1 = int(input("Enter a number: "))
    num2 = int(input("Enter another number: "))
    result = num1 / num2
    print(f"Result:{result}")
except Exception as e:
    print(f"An error occured: {e}")


class Dog:
    pass
dog= Dog()
dog.name = "Bubbles"
dog.birth_year = 2001
print(f"{dog.name} was born in {dog.birth_year}.")


class Dog:
    def __init__(self,name,birth_year):
        self.name = name
        self.birth_year = birth_year
dog = Dog("Bob",2001)
print(f"{dog.name} was born in {dog.birth_year}.")


class Dog:
    def __init__(self,name,birth_year,sound="Woof woof"):
        self.name = name
        self.birth_year = birth_year
        self.sound = sound

    def bark(self,times):
        for i in range(times):
            print(self.sound)
        return

dog1 = Dog()
dog2 = Dog("Yip yip yip")

dog1.bark(3)
dog2.bark(5)
from an import names'''



class Dog:
    created = 0
    def __init__(self,name,birth_year, sound="Woof woof"):
        self.name = name
        self.birth_year = birth_year
        self.sound = sound
        Dog.created = Dog.created + 1

dog1 = Dog("Rascal", 2012)
dog2 = Dog("Bruno", 2014, "Yip yip")
dog3 = Dog("Tommy", 2016)
dog4 = Dog("Selfie", 2018)
dog5 = Dog("Rahul", 2022)
dog6 = Dog("Raja", 2024)
print(f"{Dog.created} dogs have been cerated so far.")
