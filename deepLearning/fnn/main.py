# Extended training with multiple graphs + well-commented code

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================
# 1. LOAD DATASET
# ==============================
data = load_breast_cancer()  # Load dataset
X = data.data                # Features (input)
y = data.target              # Labels (0 or 1)

# ==============================
# 2. TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 3. FEATURE SCALING
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on train
X_test = scaler.transform(X_test)        # Apply on test

# ==============================
# 4. CONVERT TO TENSORS
# ==============================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ==============================
# 5. DEFINE FNN MODEL
# ==============================
class FNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        # Sequential model (layer by layer)
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),  # Input → Hidden
            nn.ReLU(),                  # Activation
            
            nn.Linear(64, 32),          # Hidden → Hidden
            nn.ReLU(),
            
            nn.Linear(32, 1),           # Hidden → Output
            nn.Sigmoid()                # Convert to probability
        )
    
    def forward(self, x):
        return self.net(x)

# Initialize model
model = FNN(X_train.shape[1])

# ==============================
# 6. LOSS & OPTIMIZER
# ==============================
criterion = nn.BCELoss()                    # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==============================
# 7. TRAINING LOOP
# ==============================
epochs = 50
train_losses = []
test_losses = []
accuracies = []

for epoch in range(epochs):
    # ---- TRAINING ----
    model.train()
    outputs = model(X_train)                # Forward pass
    loss = criterion(outputs, y_train)      # Compute loss
    
    optimizer.zero_grad()                  # Clear gradients
    loss.backward()                        # Backpropagation
    optimizer.step()                       # Update weights
    
    train_losses.append(loss.item())
    
    # ---- EVALUATION ----
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
        
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        accuracies.append(accuracy)

# ==============================
# 8. PLOTS
# ==============================

# Plot 1: Training Loss
plt.figure()
plt.plot(range(1, epochs+1), train_losses)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss Curve")
plt.show()

# Plot 2: Test Loss
plt.figure()
plt.plot(range(1, epochs+1), test_losses)
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.title("Test Loss Curve")
plt.show()

# Plot 3: Accuracy
plt.figure()
plt.plot(range(1, epochs+1), accuracies)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.show()

# Final Accuracy
print("Final Test Accuracy:", accuracies[-1])
