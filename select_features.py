import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the dataset
df = pd.read_csv("final_dataset.csv")

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Define selected features
selected_features = [
    'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'SYN Flag Count', 'ACK Flag Count', 'FIN Flag Count', 'RST Flag Count',
    'Fwd Header Length', 'Bwd Header Length', 'Label'
]

# Keep only the selected columns
df = df[selected_features]

# Check for NaN and infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Convert inf values to NaN
df.dropna(inplace=True)  # Drop rows with NaN values

# Separate features & labels
X = df.drop(columns=['Label'])
y = df['Label']  # 0 = Normal, 1 = Attack

# Normalize the feature data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_normalized = pd.DataFrame(X_scaled, columns=X.columns)

# Save the processed dataset
final_df = pd.concat([X_normalized, y.reset_index(drop=True)], axis=1)
final_df.to_csv("normalized_dataset.csv", index=False)

print("Feature selection & normalization complete. Saved as 'normalized_dataset.csv'")

