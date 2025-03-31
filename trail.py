import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]

# Create the plot
plt.plot(x, y, marker='o', linestyle='-', color='b', label="Line Plot")

# Labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Line Plot")

# Show legend
plt.legend()

# Display the graph
plt.show()

