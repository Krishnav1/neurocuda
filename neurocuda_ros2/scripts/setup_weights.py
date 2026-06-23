#!/usr/bin/env python3
"""
Auto-train MLP-MNIST SNN weights if not found.
Runs automatically on first startup. Takes ~30 seconds on CPU.
"""
import os, sys

WEIGHT_PATH = "/tmp/neurocuda/checkpoints/hub/mlp_mnist_snn.pt"

if os.path.exists(WEIGHT_PATH):
    print(f"[setup] Weights found: {WEIGHT_PATH}")
    sys.exit(0)

print("[setup] No weights found. Training MLP-MNIST from scratch (CPU, ~30s)...")

import torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

# Load MNIST
mnist = datasets.MNIST("/tmp/mnist_data", train=False, download=True)
images = np.array([np.array(img) for img, _ in mnist])
labels = np.array([label for _, label in mnist])

X = torch.from_numpy(images).float() / 255.0
X = X.view(-1, 784)
y = torch.from_numpy(labels).long()

# Split
n = len(X)
split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model (matches SNN architecture: Flatten → Linear→IF → Linear→IF → Linear)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x):
        return self.net(x)

model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

model.train()
for epoch in range(10):
    correct = 0
    for bx, by in train_dl:
        opt.zero_grad()
        loss = loss_fn(model(bx), by)
        loss.backward(); opt.step()
        correct += (model(bx).argmax(1) == by).sum().item()
    print(f"  Epoch {epoch+1}: acc={100*correct/len(train_ds):.1f}%")

# Test
model.eval()
with torch.no_grad():
    out = model(X_test)
    test_acc = (out.argmax(1) == y_test).sum().item() / len(y_test) * 100
print(f"  Test accuracy: {test_acc:.1f}%")

# Convert to SNN keys
state = model.state_dict()
snn_state = {}
mapping = [
    ("net.1.weight", "fc1.weight"), ("net.1.bias", "fc1.bias"),
    ("net.3.weight", "fc2.weight"), ("net.3.bias", "fc2.bias"),
    ("net.5.weight", "fc3.weight"), ("net.5.bias", "fc3.bias"),
]
for old, new in mapping:
    snn_state[new] = state[old]
snn_state["if1.thresh"] = torch.tensor(1.0)
snn_state["if2.thresh"] = torch.tensor(1.0)

os.makedirs(os.path.dirname(WEIGHT_PATH), exist_ok=True)
torch.save(snn_state, WEIGHT_PATH)
print(f"[setup] Weights saved: {WEIGHT_PATH}  (accuracy: {test_acc:.1f}%)")
