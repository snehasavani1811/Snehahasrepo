
seasons = ("Winter", "Spring", "Summer", "Autumn")
month = int(input("Enter the number of the month (1-12): "))
if month in [12, 1, 2]:
    season = seasons[0]  # Winter
elif month in [3, 4, 5]:
    season = seasons[1]  # Spring
elif month in [6, 7, 8]:
    season = seasons[2]  # Summer
elif month in [9, 10, 11]:
    season = seasons[3]  # Autumn
else:
    season = "Invalid month number"
print(f"The season is: {season}")



names_set = set()
while True:
    name = input("Enter a name (or press Enter to stop): ")
    if name == "":
        break
    if name in names_set:
        print("Existing name")
    else:
        print("New name")
        names_set.add(name)
print("\nNames entered:")
for name in names_set:
    print(name)



    airport_data = {}

    while True:
        # Display the menu of options
        print("\nMenu:")
        print("1. Enter a new airport")
        print("2. Fetch airport information")
        print("3. Quit")
        choice = input("Choose an option (1-3): ")
        if choice == "1":
            icao_code = input("Enter the ICAO code of the airport: ").upper()  # Ensure ICAO code is uppercase
            airport_name = input("Enter the name of the airport: ")
            if icao_code in airport_data:
                print("This ICAO code is already in the system.")
            else:
                airport_data[icao_code] = airport_name
                print(f"Airport {airport_name} with ICAO code {icao_code} added.")
        elif choice == "2":
            icao_code = input("Enter the ICAO code of the airport to fetch: ").upper()
            if icao_code in airport_data:
                print(f"The airport name for ICAO code {icao_code} is: {airport_data[icao_code]}")
            else:
                print("No airport found with that ICAO code.")
        elif choice == "3":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")



