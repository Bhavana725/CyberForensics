import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ==========================
# 🚀 STEP 1: LOAD DATASET
# ==========================
df = pd.read_csv("10k_dataset.csv")

# Fix column names (strip spaces)
df.columns = df.columns.str.strip()

# Handle NaN and Infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)  # Removes rows with NaN values

# Encode categorical labels if needed
if 'Label' in df.columns:
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])

# Separate features and target
X = df.drop(columns=['Label'])
y = df['Label']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to NumPy arrays
X_scaled = np.array(X_scaled, dtype=np.float32)  # Ensuring float32 for PyTorch
y = np.array(y, dtype=np.int64)  # Ensure proper integer format for labels

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

# =================================
# 🚀 STEP 2: DEFINE RL DQN MODEL
# =================================
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

# Define Device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Model
input_dim = X_train.shape[1]  # Number of features
output_dim = len(np.unique(y))  # Number of classes
dqn = DQN(input_dim, output_dim).to(device)

# Optimizer & Loss Function
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# ==========================
# 🚀 STEP 3: TRAIN MODEL
# ==========================
# Convert Data to Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

# Training Loop
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    dqn.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = dqn(batch_X)
        loss = loss_fn(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save Model
torch.save(dqn.state_dict(), "dqn_ddos_model.pth")
print("✅ Model Training Complete and Saved.")

# ==========================
# 🚀 STEP 4: TEST MODEL
# ==========================
def predict_threat(instance):
    instance = torch.tensor(instance, dtype=torch.float32).to(device).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = dqn(instance)
        prediction = torch.argmax(output).item()
    return prediction

# Test on 5 random samples
for i in range(5):
    test_instance = X_test[i]
    predicted_label = predict_threat(test_instance)
    print(f"🔹 Test {i+1}: Predicted Threat Level: {predicted_label}")

# ==========================
# 🚀 STEP 5: EVALUATE MODEL
# ==========================
def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predicted_labels = torch.argmax(predictions, dim=1)  # Get the class with highest probability
        accuracy = (predicted_labels == y_test_tensor).sum().item() / y_test_tensor.size(0)
    return accuracy

# Compute accuracy
test_accuracy = evaluate_model(dqn, X_test_tensor, y_test_tensor)
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")

