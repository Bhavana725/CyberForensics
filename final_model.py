import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Load Dataset
df = pd.read_csv("10k_dataset.csv")
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Encode Labels
if 'Label' in df.columns:
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])

# Separate Features and Target
X = df.drop(columns=['Label'])
y = df['Label']

# Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute Feature Importance using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
importances = rf.feature_importances_

# Sort features by importance
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Select Top 9 Features
top_9_features = ['Init_Win_bytes_forward', 'Fwd Packet Length Max', 'Subflow Fwd Bytes',
                  'Bwd Packet Length Max', 'Fwd Packet Length Mean', 'Avg Fwd Segment Size',
                  'Avg Bwd Segment Size', 'Total Length of Fwd Packets', 'Destination Port']

# Train using only Top 9 Features
X_selected = df[top_9_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Convert Data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Define Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN(input_dim=9, output_dim=len(np.unique(y))).to(device)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

# Train Model
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    dqn.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = dqn(batch_X.to(device))
        loss = loss_fn(predictions, batch_y.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save Model
torch.save(dqn.state_dict(), "f_model_top9.pth")
print("✅ Model Trained and Saved as model_top9.pth")

# Evaluate Model
def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor.to(device))
        predicted_labels = torch.argmax(predictions, dim=1)

        # Convert to NumPy for Metrics
        y_true = y_test_tensor.cpu().numpy()
        y_pred = predicted_labels.cpu().numpy()

        # Compute Metrics
        accuracy = np.mean(y_pred == y_true) * 100
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)

        print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
        print("\n📊 Classification Report:")
        print(pd.DataFrame(report).transpose())

        # Save Confusion Matrix Image
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")  # Saves image
        print("\n✅ Confusion Matrix saved as 'confusion_matrix.png'")

    return accuracy, report, "confusion_matrix.png"

# Compute and Save Results
test_accuracy, class_report, conf_matrix_path = evaluate_model(dqn, X_test_tensor, y_test_tensor)

# Final Output for Dashboard
print("\n📌 Use the following details for your dashboard:")
print(f"➡️ Accuracy: {test_accuracy:.2f}%")
print(f"➡️ Confusion Matrix Path: {conf_matrix_path}")
print(f"➡️ Precision, Recall, F1-Score:\n{pd.DataFrame(class_report).transpose()}")


