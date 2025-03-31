import pandas as pd
import matplotlib.pyplot as plt

data = {
    1: 78.35, 2: 85.95, 3: 85.50, 4: 87.30, 5: 81.25, 6: 87.70, 7: 99.70, 8: 98.95, 9: 99.20, 10: 99.50,
    11: 99.30, 12: 99.45, 13: 99.50, 14: 99.50, 15: 99.50, 16: 99.75, 17: 99.75, 18: 99.75, 19: 99.75, 20: 99.80,
    21: 99.80, 22: 99.80, 23: 99.80, 24: 99.75, 25: 99.85, 26: 99.85, 27: 99.85, 28: 99.90, 29: 99.85, 30: 99.80,
    31: 99.95, 32: 99.75, 33: 99.75, 34: 99.90, 35: 99.75, 36: 99.85, 37: 99.70, 38: 99.85, 39: 99.70, 40: 99.90,
    41: 99.65, 42: 99.85, 43: 99.80, 44: 99.85, 45: 99.45, 46: 99.95, 47: 99.90, 48: 99.90, 49: 99.90, 50: 99.90,
    51: 99.80, 52: 99.85, 53: 99.85, 54: 99.65, 55: 99.85, 56: 99.90, 57: 99.70, 58: 99.85, 59: 99.55, 60: 99.85,
    61: 99.55, 62: 99.75, 63: 99.90, 64: 99.90, 65: 99.90, 66: 99.90, 67: 99.90, 68: 99.80, 69: 99.90, 70: 99.90,
    71: 99.85, 72: 99.90, 73: 99.90, 74: 99.90, 75: 99.85, 76: 99.90, 77: 99.85, 78: 99.90
}

multiples_of_5 = [i for i in range(5, 80, 5)]
multiples_of_6 = [i for i in range(6, 79, 6)]

accuracy_5 = [data[i] for i in multiples_of_5]
accuracy_6 = [data[i] for i in multiples_of_6]

# Table Creation
table_5 = pd.DataFrame({'Features (Multiples of 5)': multiples_of_5, 'Accuracy (%)': accuracy_5})
table_6 = pd.DataFrame({'Features (Multiples of 6)': multiples_of_6, 'Accuracy (%)': accuracy_6})

print("Table for Multiples of 5 Features:")
print(table_5)

print("\nTable for Multiples of 6 Features:")
print(table_6)

# Graph Creation
plt.figure(figsize=(10, 5))
plt.plot(multiples_of_5, accuracy_5, marker='o', linestyle='-', color='blue')
plt.title('Accuracy Trend (Multiples of 5 Features)')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.savefig('accuracy_trend_5.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(multiples_of_6, accuracy_6, marker='o', linestyle='-', color='green')
plt.title('Accuracy Trend (Multiples of 6 Features)')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.savefig('accuracy_trend_6.png')
plt.show()
