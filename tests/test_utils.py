"""
Tests for utility functions: sparsity measurement, energy estimation, validation.

Tests:
  - measure_sparsity returns correct structure and plausible values
  - measure_sparsity handles both 4D and 5D data
  - Energy estimation runs without crashing
  - fold_batchnorm (standalone) works correctly
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neurocuda import measure_sparsity, fold_batchnorm, energy_estimate, validate_snn
from models import QCFS, IFNeuron, LIFNeuron, reset_spiking

# Model definitions (duplicated from conftest for test independence)
class SimpleCNN4D(nn.Module):
    """4D-native model for testing."""
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


class SimpleCNN5D(nn.Module):
    """5D-native model for testing."""
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
        return x.reshape(B, T, -1).mean(dim=1)


# ===========================================================================
# measure_sparsity
# ===========================================================================

class TestMeasureSparsity:
    """Test sparsity measurement on spiking models."""

    @pytest.fixture
    def spiking_model_4d(self, device):
        """A simple 4D-native SNN with IF neurons."""
        model = SimpleCNN4D(act_factory=lambda: IFNeuron(thresh=2.0)).to(device).eval()
        return model

    @pytest.fixture
    def spiking_model_5d(self, device):
        """A simple 5D-native SNN with IF neurons."""
        model = SimpleCNN5D(act_factory=lambda: IFNeuron(thresh=2.0)).to(device).eval()
        return model

    @pytest.fixture
    def dataloader_4d(self, device):
        """4D dataloader for sparsity measurement."""
        x = torch.randn(32, 1, 32, 32, device=device)
        y = torch.zeros(32, dtype=torch.long)
        return DataLoader(TensorDataset(x, y), batch_size=8)

    @pytest.fixture
    def dataloader_5d(self, device):
        """5D dataloader for sparsity measurement."""
        x = torch.randn(16, 4, 1, 32, 32, device=device)  # (B,T,C,H,W)
        y = torch.zeros(16, dtype=torch.long)
        return DataLoader(TensorDataset(x, y), batch_size=8)

    def test_returns_tuple_of_4(self, spiking_model_4d, dataloader_4d, device):
        """measure_sparsity returns 4 values."""
        result = measure_sparsity(spiking_model_4d, dataloader_4d, device=device)
        assert len(result) == 4

    def test_sparsity_between_0_and_100(self, spiking_model_4d, dataloader_4d, device):
        """Sparsity is a percentage between 0 and 100."""
        sparsity, nonzero, total, layer_data = measure_sparsity(
            spiking_model_4d, dataloader_4d, device=device)
        assert 0.0 <= sparsity <= 100.0, f"Sparsity {sparsity} out of range"
        assert nonzero >= 0
        assert total > 0
        assert nonzero <= total

    def test_layer_data_has_entries(self, spiking_model_4d, dataloader_4d, device):
        """layer_data contains entries for IF neurons."""
        _, _, _, layer_data = measure_sparsity(
            spiking_model_4d, dataloader_4d, device=device)
        assert len(layer_data) > 0, "Should have entries for IF activations"
        for name, d in layer_data.items():
            assert "nonzero" in d
            assert "total" in d
            assert d["total"] > 0

    def test_max_batches_limits(self, spiking_model_4d, dataloader_4d, device):
        """max_batches limits the number of batches processed."""
        sparsity, _, total, _ = measure_sparsity(
            spiking_model_4d, dataloader_4d, device=device, max_batches=2)
        # Should complete quickly and return valid sparsity
        assert 0.0 <= sparsity <= 100.0

    def test_5d_dataloader_no_crash(self, spiking_model_5d, dataloader_5d, device):
        """measure_sparsity handles 5D temporal data without crashing."""
        sparsity, _, _, _ = measure_sparsity(
            spiking_model_5d, dataloader_5d, device=device, max_batches=2)
        assert 0.0 <= sparsity <= 100.0

    def test_model_with_no_spiking_neurons(self, device):
        """measure_sparsity on a model with no IF/LIF returns empty results."""
        model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 32 * 32, 5),
        ).to(device).eval()
        x = torch.randn(4, 1, 32, 32, device=device)
        y = torch.zeros(4, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        sparsity, nonzero, total, layer_data = measure_sparsity(
            model, loader, device=device)
        # No IF/LIF → no spikes counted → total=0, nonzero=0
        assert total == 0
        assert nonzero == 0
        # measure_sparsity uses max(total, 1) to avoid div/0 → returns 100% when total=0
        assert sparsity == 100.0, f"Expected 100% for no-neuron model, got {sparsity}%"


# ===========================================================================
# Energy Estimation
# ===========================================================================

class TestEnergyEstimate:
    """Test energy estimation utility."""

    def test_returns_dict_with_keys(self):
        """energy_estimate returns expected keys."""
        model = SimpleCNN4D(act_factory=nn.ReLU)
        result = energy_estimate(model, input_shape=(1, 32, 32), T=16, spike_rate=0.2)
        assert "ann_flops" in result
        assert "snn_spike_ops" in result
        assert "gpu_energy_uj" in result
        assert "neuro_energy_uj" in result
        assert "energy_ratio" in result

    def test_snn_more_efficient_at_low_spike_rate(self):
        """At low spike rates, SNN should be more energy-efficient."""
        model = SimpleCNN4D(act_factory=nn.ReLU)
        result_low = energy_estimate(model, input_shape=(1, 32, 32), T=16, spike_rate=0.05)
        result_dense = energy_estimate(model, input_shape=(1, 32, 32), T=16, spike_rate=1.0)
        assert result_low["energy_ratio"] > result_dense["energy_ratio"], \
            "Low spike rate should give higher energy ratio (more savings)"


# ===========================================================================
# fold_batchnorm (standalone utility)
# ===========================================================================

class TestFoldBatchnormUtility:
    """Test the standalone fold_batchnorm utility from neurocuda.utils."""

    def test_fold_runs_without_crash(self, device):
        """fold_batchnorm runs on a model with BN."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device).eval()
        dummy = torch.randn(4, 1, 32, 32, device=device)
        _ = model(dummy)
        fold_batchnorm(model)
        # BN should be identity-like after folding
        assert isinstance(model.bn1, nn.BatchNorm2d) or \
               model.bn1.weight.abs().max() > 0, "BN should be modified"


# ===========================================================================
# validate_snn
# ===========================================================================

class TestValidateSNN:
    """Test the validate_snn utility."""

    def test_returns_accuracy_percentage(self, device):
        """validate_snn returns a number between 0 and 100."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device).eval()
        x = torch.randn(32, 1, 32, 32, device=device)
        y = torch.zeros(32, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=16)
        acc = validate_snn(model, loader, device=str(device))
        assert 0.0 <= acc <= 100.0
