import pandas as pd

# Load the dataset
df = pd.read_csv("final_dataset.csv")

# Sample 10,000 instances randomly
df_sampled = df.sample(n=10000, random_state=42)

# Save the new dataset
df_sampled.to_csv("10k_dataset.csv", index=False)

# Print first 5 instances
print(df_sampled.head())

