import pandas as pd

file_path = "/home/bhavana/Downloads/dataset/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(file_path)

# Encoding 'Label' column: BENIGN → 0, DDoS → 1
df[' Label'] = df[' Label'].replace({'BENIGN': 0, 'DDoS': 1})

# Save the modified dataset (optional)
df.to_csv("encoded_dataset.csv", index=False)

# Check if encoding is successful
print(df[' Label'].value_counts())  

