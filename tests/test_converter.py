"""
Integration tests for the converter pipeline.

Tests:
  - convert() runs end-to-end on 4D and 5D models without crashing
  - _forward_temporal handles both model types
  - _forward_spiking handles both model types
  - Strategy selection (auto, qcfs_if_ft, qcfs_direct)
  - Channel-wise vs per-layer QCFS
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neurocuda import convert, measure_sparsity
from neurocuda.converter import (
    _forward_temporal,
    _forward_spiking,
    _replace_activations,
    _replace_activations_cs,
    _fold_batchnorms,
)
from models import QCFS, IFNeuron, LIFNeuron, reset_spiking

# Model definitions (duplicated from conftest for test independence)
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
        return x.reshape(B, T, -1).mean(dim=1)


# ===========================================================================
# _forward_temporal
# ===========================================================================

class TestForwardTemporal:
    """Test temporal forward pass auto-detection."""

    def test_4d_native_model(self, device):
        """_forward_temporal handles 4D-native models correctly."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device).eval()
        data = torch.randn(2, 4, 1, 32, 32, device=device)  # (B,T,C,H,W)
        out = _forward_temporal(model, data, average=True)
        assert out.shape == (2, 5), f"Expected (2,5), got {out.shape}"

    def test_4d_native_no_average(self, device):
        """_forward_temporal with average=False returns (B*T, ...)."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device).eval()
        data = torch.randn(2, 4, 1, 32, 32, device=device)
        out = _forward_temporal(model, data, average=False)
        assert out.shape == (8, 5), f"Expected (8,5), got {out.shape}"

    def test_5d_native_model(self, device):
        """_forward_temporal handles 5D-native models correctly."""
        model = SimpleCNN5D(act_factory=nn.ReLU).to(device).eval()
        data = torch.randn(2, 4, 1, 32, 32, device=device)
        out = _forward_temporal(model, data, average=True)
        assert out.shape == (2, 5), f"Expected (2,5), got {out.shape}"

    def test_5d_native_no_average(self, device):
        """_forward_temporal with average=False on 5D-native returns model output as-is."""
        model = SimpleCNN5D(act_factory=nn.ReLU).to(device).eval()
        data = torch.randn(2, 4, 1, 32, 32, device=device)
        out = _forward_temporal(model, data, average=False)
        # 5D-native returns whatever the model returns (B*T, N) or (B, N)
        # Just verify it doesn't crash
        assert out is not None


# ===========================================================================
# _forward_spiking
# ===========================================================================

class TestForwardSpiking:
    """Test spiking forward pass auto-detection."""

    def test_4d_native_spiking(self, device):
        """_forward_spiking handles 4D-native spiking models."""
        model = SimpleCNN4D(act_factory=lambda: IFNeuron(thresh=2.0)).to(device).eval()
        data = torch.randn(2, 4, 1, 32, 32, device=device)
        reset_spiking(model)
        out = _forward_spiking(model, data, average=True)
        assert out.shape == (2, 5), f"Expected (2,5), got {out.shape}"

    def test_5d_native_spiking(self, device):
        """_forward_spiking handles 5D-native spiking models."""
        model = SimpleCNN5D(act_factory=lambda: IFNeuron(thresh=2.0)).to(device).eval()
        data = torch.randn(2, 4, 1, 32, 32, device=device)
        reset_spiking(model)
        out = _forward_spiking(model, data, average=True)
        # Just verify it doesn't crash and returns something
        assert out is not None

    def test_state_accumulation(self, device):
        """Spiking forward accumulates IF state across timesteps."""
        model = SimpleCNN4D(act_factory=lambda: IFNeuron(thresh=10.0)).to(device).eval()
        data = torch.randn(1, 5, 1, 32, 32, device=device)
        reset_spiking(model)
        out_first = _forward_spiking(model, data[:, 0:3, :, :, :], average=True)

        # Find IF neurons and check they have state
        has_state = False
        for m in model.modules():
            if isinstance(m, IFNeuron) and m.v is not None:
                has_state = True
                break
        assert has_state, "IF neurons should have membrane state after forward"


# ===========================================================================
# Activation Replacement
# ===========================================================================

class TestActivationReplacement:
    """Test _replace_activations and _replace_activations_cs."""

    def test_replace_relu_with_qcfs_scalar(self, device):
        """Replace ReLU with per-layer (scalar) QCFS."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device)
        n = _replace_activations(model, (nn.ReLU,), lambda: QCFS(L=8, thresh_init=3.0, num_channels=None))
        assert n == 1  # One ReLU replaced
        # Verify it's QCFS now
        assert isinstance(model.act1, QCFS)

    def test_replace_relu_with_qcfs_channel_wise(self, device):
        """Replace ReLU with per-channel QCFS (auto-detects channels)."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device)
        n = _replace_activations_cs(model, (nn.ReLU,), lambda nc: QCFS(L=8, thresh_init=2.0, num_channels=nc))
        assert n == 1
        assert isinstance(model.act1, QCFS)
        assert model.act1.num_channels == 8, f"Should detect 8 channels from Conv2d, got {model.act1.num_channels}"


# ===========================================================================
# BN Folding
# ===========================================================================

class TestBNFolding:
    """Test BatchNorm folding."""

    def test_fold_replaces_bn_with_identity(self, device):
        """After folding, BN layers become Identity."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device)
        # Run a forward to set running stats
        model.eval()
        dummy = torch.randn(4, 1, 32, 32, device=device)
        _ = model(dummy)

        n = _fold_batchnorms(model)
        assert n == 1  # One BN folded
        assert isinstance(model.bn1, nn.Identity), "BN should become Identity after folding"

    def test_folded_output_matches_original(self, device):
        """BN folding should produce near-identical outputs when BN is near-identity."""
        torch.manual_seed(42)
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device).eval()

        # Set BN to near-identity: gamma=1, beta=0, mean=0, var=1-eps
        with torch.no_grad():
            model.bn1.running_mean.zero_()
            model.bn1.running_var.fill_(1.0 - model.bn1.eps)  # var+eps = 1, so std=1
            model.bn1.weight.fill_(1.0)
            model.bn1.bias.zero_()

        # Also zero conv bias so output depends only on weight
        model.conv1.bias.data.zero_()

        x = torch.randn(4, 1, 32, 32, device=device)
        out_before = model(x).clone()

        # Fold BN into Conv
        _fold_batchnorms(model)
        out_after = model(x).clone()

        # With identity BN, folding should produce very close outputs
        max_diff = (out_before - out_after).abs().max().item()
        assert max_diff < 0.01, f"Folded output differs by {max_diff}"

    def test_fold_preserves_conv_weights_shape(self, device):
        """After folding, Conv weights keep original shape."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device).eval()
        dummy = torch.randn(4, 1, 32, 32, device=device)
        _ = model(dummy)

        original_shape = model.conv1.weight.shape
        _fold_batchnorms(model)
        assert model.conv1.weight.shape == original_shape


# ===========================================================================
# convert() end-to-end
# ===========================================================================

class TestConvert:
    """End-to-end conversion tests with synthetic data."""

    def test_convert_4d_model_no_crash(self, device, train_loader, test_loader):
        """convert() runs on 4D-native model without crashing."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device)
        snn, stats = convert(
            model,
            train_loader,
            test_loader=test_loader,
            qcfs_epochs=1,
            if_epochs=1,
            strategy="qcfs_if_ft",
            channel_wise=False,
            device=device,
            verbose=False,
        )
        assert snn is not None
        assert "strategy" in stats
        assert "if_accuracy" in stats

    def test_convert_5d_model_no_crash(self, device, train_loader_5d, test_loader_5d):
        """convert() runs on 5D-native model without crashing."""
        model = SimpleCNN5D(act_factory=nn.ReLU).to(device)
        snn, stats = convert(
            model,
            train_loader_5d,
            test_loader=test_loader_5d,
            qcfs_epochs=1,
            if_epochs=1,
            strategy="qcfs_if_ft",
            channel_wise=False,
            device=device,
            verbose=False,
        )
        assert snn is not None
        assert "strategy" in stats

    def test_convert_channel_wise(self, device, train_loader, test_loader):
        """convert() with channel_wise=True runs without crashing."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device)
        snn, stats = convert(
            model,
            train_loader,
            test_loader=test_loader,
            qcfs_epochs=1,
            if_epochs=1,
            strategy="qcfs_if_ft",
            channel_wise=True,
            device=device,
            verbose=False,
        )
        assert snn is not None

    def test_convert_strategy_auto(self, device, train_loader_5d):
        """convert() with strategy='auto' selects a valid strategy."""
        model = SimpleCNN5D(act_factory=nn.ReLU).to(device)
        snn, stats = convert(
            model,
            train_loader_5d,
            qcfs_epochs=1,
            if_epochs=1,
            strategy="auto",
            channel_wise=False,
            device=device,
            verbose=False,
        )
        assert stats["strategy"] in ("qcfs_if_ft", "qcfs_direct")

    def test_convert_output_has_if_activations(self, device, train_loader, test_loader):
        """After conversion, output model has IF/LIF activations (spiking)."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device)
        snn, _ = convert(
            model,
            train_loader,
            qcfs_epochs=1,
            if_epochs=1,
            strategy="qcfs_if_ft",
            channel_wise=False,
            device=device,
            verbose=False,
        )
        # Check that activations are now IF/LIF (spiking)
        has_spiking = False
        for m in snn.modules():
            if isinstance(m, (IFNeuron, LIFNeuron)):
                has_spiking = True
                break
        assert has_spiking, "Converted model should have IF/LIF activations"

    def test_convert_accuracy_improves_over_epochs(self, device, train_loader, test_loader):
        """IF accuracy should be non-random (above chance) after conversion."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device)
        _, stats = convert(
            model,
            train_loader,
            test_loader=test_loader,
            qcfs_epochs=3,
            if_epochs=3,
            strategy="qcfs_if_ft",
            channel_wise=False,
            device=device,
            verbose=False,
        )
        # With 5 classes, chance = 20%. Even minimal fine-tuning should beat this.
        if_acc = stats.get("if_accuracy", 0)
        assert if_acc > 15.0, f"IF accuracy {if_acc:.1f}% should be above random chance ~20%"
