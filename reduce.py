import pandas as pd

# Load the encoded dataset
df = pd.read_csv("encoded_dataset.csv")

# Separate the two classes
df_0 = df[df[' Label'] == 0]  # BENIGN
df_1 = df[df[' Label'] == 1]  # DDoS

# Undersample the majority class (DDoS)
df_1 = df_1.sample(n=97000, random_state=42)
df_0 = df_0.sample(n=97000, random_state=42)  # Keep 97000 from BENIGN as well

# Combine the balanced dataset
df_balanced = pd.concat([df_0, df_1])

# Save the new dataset
df_balanced.to_csv("final_dataset.csv", index=False)

# Check final label distribution
print(df_balanced[' Label'].value_counts())

