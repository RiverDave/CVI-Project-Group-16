import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # 1. Generate sample data (e.g., from a normal distribution)
# data = np.random.randn(1000)


dataset = pd.read_csv('data/driving_log.csv')
steering_angle = dataset.iloc[:, 3]

plt.hist(steering_angle, bins=30, color='skyblue', edgecolor='black')

# # 3. Add labels and a title
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Sample Data")

# # 4. Display the plot
plt.show()
