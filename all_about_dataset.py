import pandas as pd

# Load the dataset
df = pd.read_csv("final_dataset.csv")

# Print dataset shape (rows, columns)
print(f"Total Instances: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")

# Print feature names
print("\nFeature Names:")
print(df.columns.tolist())

# Print data types of each feature
print("\nFeature Data Types:")
print(df.dtypes)

# Print class distribution (if a label column exists)
if "label" in df.columns or "target" in df.columns:
    class_col = "label" if "label" in df.columns else "target"
    print("\nClass Distribution:")
    print(df[class_col].value_counts())

