import pandas as pd

# Load the dataset (replace 'dataset.csv' with your actual file)
df = pd.read_csv("10k_dataset.csv")

# Print the feature names (column names)
print("Feature Names:")
print(df.columns.tolist())

