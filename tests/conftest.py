"""
Shared fixtures for NeuroCUDA tests.

All tests use synthetic data — no downloads, no pretrained checkpoints.
Tests should run in <30 seconds total on CPU, <10 seconds on CUDA.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ===========================================================================
# Device
# ===========================================================================

@pytest.fixture(scope="session")
def device():
    """Use CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Simple 4D-native CNN (expects B,C,H,W)
# ===========================================================================

class SimpleCNN4D(nn.Module):
    """4D-native model: forward expects (B, C, H, W)."""
    def __init__(self, act_factory=nn.ReLU, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.act1 = act_factory()
        self.pool = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool(self.act1(self.bn1(self.conv1(x))))
        return self.fc(self.flatten(x))


# ===========================================================================
# Simple 5D-native CNN (expects B,T,C,H,W)
# ===========================================================================

class SimpleCNN5D(nn.Module):
    """5D-native model: forward expects (B, T, C, H, W), does own reshape."""
    def __init__(self, act_factory=nn.ReLU, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.act1 = act_factory()
        self.pool = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 16 * 16, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.pool(self.act1(self.bn1(self.conv1(x))))
        x = self.flatten(x)
        x = self.fc(x)
        return x.reshape(B, T, -1).mean(dim=1)  # temporal average


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="function")
def model_4d():
    """Fresh 4D-native CNN with ReLU."""
    return SimpleCNN4D(act_factory=nn.ReLU)


@pytest.fixture(scope="function")
def model_5d():
    """Fresh 5D-native CNN with ReLU."""
    return SimpleCNN5D(act_factory=nn.ReLU)


@pytest.fixture(scope="function")
def synthetic_4d_data(device):
    """Synthetic 4D data: (B, C, H, W) with 2 classes."""
    x = torch.randn(16, 1, 32, 32, device=device)
    y = torch.randint(0, 5, (16,), device=device)
    return x, y


@pytest.fixture(scope="function")
def synthetic_5d_data(device):
    """Synthetic 5D data: (B, T, C, H, W) with 2 classes."""
    x = torch.randn(8, 4, 1, 32, 32, device=device)
    y = torch.randint(0, 5, (8,), device=device)
    return x, y


@pytest.fixture(scope="function")
def train_loader(synthetic_4d_data):
    """Synthetic DataLoader for calibration/fine-tuning."""
    x, y = synthetic_4d_data
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)


@pytest.fixture(scope="function")
def test_loader():
    """Synthetic DataLoader for testing."""
    x = torch.randn(32, 1, 32, 32)
    y = torch.randint(0, 5, (32,))
    return DataLoader(TensorDataset(x, y), batch_size=16)


@pytest.fixture(scope="function")
def train_loader_5d():
    """Synthetic 5D DataLoader."""
    x = torch.randn(16, 4, 1, 32, 32)  # (B, T, C, H, W)
    y = torch.randint(0, 5, (16,))
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)


@pytest.fixture(scope="function")
def test_loader_5d():
    """Synthetic 5D DataLoader for testing."""
    x = torch.randn(16, 4, 1, 32, 32)
    y = torch.randint(0, 5, (16,))
    return DataLoader(TensorDataset(x, y), batch_size=8)
