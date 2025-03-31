import pandas as pd

file_path = "/home/bhavana/Downloads/dataset/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(file_path)

# Print column names and the count of instances
print("Columns:", list(df.columns))  
print("Total Instances:", len(df))  # Count of rows

