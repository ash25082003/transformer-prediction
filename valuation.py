import os
import matplotlib.pyplot as plt

def read_numbers_from_file(file_path):
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file]
    return numbers

def batch_average(numbers, batch_size=1500):
    # Number of batches needed
    num_batches = (len(numbers) + batch_size - 1) // batch_size
    
    # List to hold the average of each batch
    batch_averages = []
    
    for i in range(num_batches):
        # Extract the batch
        batch = numbers[i * batch_size:(i + 1) * batch_size]
        # Calculate the average of the batch
        batch_avg = sum(batch) / len(batch)
        # Append the average to the list
        batch_averages.append(batch_avg)
    
    return batch_averages

# Path to the text file
file_path = 'losses.txt'

# Read numbers from the file
numbers = read_numbers_from_file(file_path)

# Calculate the averages in batches of 100
averages = batch_average(numbers)
print(averages)

# Plot the result
plt.plot(averages)
plt.xlabel('Batch Number')
plt.ylabel('Average Loss')
plt.title('Average Loss per Batch')
plt.show()
