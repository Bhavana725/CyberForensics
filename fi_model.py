import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# ==========================
# 🚀 STEP 1: LOAD DATASET
# ==========================
df = pd.read_csv("10k_dataset.csv")
df.columns = df.columns.str.strip()  # Fix column names
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Encode target labels
if 'Label' in df.columns:
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])

# Separate features and target
X = df.drop(columns=['Label'])
y = df['Label']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.array(X_scaled, dtype=np.float32)  # Ensure float32 for PyTorch
y = np.array(y, dtype=np.int64)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========================
# 🚀 STEP 2: FEATURE IMPORTANCE
# ==========================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importance = rf.feature_importances_

# Rank features
feature_ranking = sorted(zip(X.columns, feature_importance), key=lambda x: x[1], reverse=True)
ranked_features = [feat[0] for feat in feature_ranking]

print("🔹 Feature Importance Ranking:")
for rank, (feat, score) in enumerate(feature_ranking, start=1):
    print(f"{rank}. {feat} - Importance: {score:.5f}")

# ==========================
# 🚀 STEP 3: DEFINE DQN MODEL
# ==========================
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dim = len(np.unique(y))

# ==========================
# 🚀 STEP 4: TRAIN WITH INCREASING FEATURES
# ==========================
accuracies = {}

for i in range(1, len(ranked_features) + 1):
    selected_features = ranked_features[:i]
    X_selected = df[selected_features].values
    X_selected_scaled = scaler.fit_transform(X_selected)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected_scaled, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    dqn = DQN(i, output_dim).to(device)
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    
    # Training Loop
    for epoch in range(5):  # Train for 5 epochs for each subset
        dqn.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = dqn(batch_X)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    dqn.eval()
    with torch.no_grad():
        predictions = dqn(X_test_tensor)
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = (predicted_labels == y_test_tensor).sum().item() / y_test_tensor.size(0)
    
    accuracies[i] = accuracy
    print(f"✅ Features Used: {i} | Test Accuracy: {accuracy * 100:.2f}%")

# ==========================
# 🚀 STEP 5: FINAL RESULTS
# ==========================
print("\n🔹 Accuracy Trend with Increasing Features:")
for feat_count, acc in accuracies.items():
    print(f"{feat_count} features: {acc * 100:.2f}% accuracy")

