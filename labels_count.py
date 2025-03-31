import pandas as pd

file_path = "/home/bhavana/Downloads/dataset/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(file_path)

# Print unique values and their count in the 'Label' column
print(df[' Label'].value_counts())  

