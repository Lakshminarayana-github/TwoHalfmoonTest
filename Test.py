import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# ----------------------------
# Load Test Dataset
# ----------------------------
np.random.seed(42)
torch.manual_seed(42)

data_test = pd.read_csv('2halfmoonsTest.csv')  # replace with your test dataset

print("Test Dataset Information:")
print(f"Shape: {data_test.shape}")
print(f"\nClass distribution:")
print(data_test['ClassLabel'].value_counts())

X_test = data_test[['X', 'Y']].values
y_test = data_test['ClassLabel'].values
y_test = y_test - 1  # remap to 0/1

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ----------------------------
# Define the same MLP architecture
# ----------------------------
class MLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=16, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# ----------------------------
# Load pre-trained model
# ----------------------------
model = MLP(input_size=2, hidden_size=16, output_size=1)
model.load_state_dict(torch.load('mlp_halfmoons.pth'))  # your saved model path
model.eval()

# ----------------------------
# Predictions
# ----------------------------
with torch.no_grad():
    outputs_test = model(X_test_tensor)
    predictions_test = (outputs_test >= 0.5).int().numpy().flatten()

y_true = y_test
y_pred = predictions_test

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum()

print("\nConfusion Matrix (counts):")
print(cm)
print("\nConfusion Matrix (normalized %):")
print(np.round(cm_normalized * 100, 2))

# ----------------------------
# True Positives, False Positives, etc.
# ----------------------------
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

print(f"\nTrue Positives:  {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"True Negatives:  {TN}")

# ----------------------------
# Metrics
# ----------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nEvaluation Metrics:")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1-score : {f1*100:.2f}%")

# ----------------------------
# Optional: Plot Confusion Matrix
# ----------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False)
plt.title("Confusion Matrix (Counts)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(cm_normalized*100, annot=True, fmt=".2f", cmap='Blues', cbar=False)
plt.title("Confusion Matrix (Normalized %)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
