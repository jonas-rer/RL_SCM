episode = 30
length = 30
lead_time = 3
done = False

planned_demand = [0, 0, 0, 0, 0, 15,
                0, 0, 0, 0, 0, 15,
                0, 0, 0, 0, 0, 15,
                0, 0, 0, 0, 0, 15,
                0, 0, 0, 0, 0, 15,
                0, 0, 0, 0, 0, 15]

expected_demand = 0

while done == False:

    # Get the current day
    i = length - episode

    if i + lead_time < len(planned_demand):
        expected_demand = planned_demand[i+lead_time]
        print(f"Expected demand for day {i+lead_time+1} is {expected_demand}")
    else:
        print("Index out of range")

    episode -= 1

    # Check if episode is done
    if episode <= 0: 
        done = True
    else:
        done = False