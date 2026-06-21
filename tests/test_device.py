"""
Device placement tests — verify all parameters end up on the correct device.

These tests validate the fixes from June 21, 2026:
  - QCFS parameters on correct device after _replace_activations_cs
  - IF parameters on correct device after _transfer_qcfs_to_if
  - Full convert() pipeline keeps everything on target device
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neurocuda import convert
from neurocuda.converter import (
    _replace_activations,
    _replace_activations_cs,
    _transfer_qcfs_to_if,
    _fold_batchnorms,
)
from models import QCFS, IFNeuron, LIFNeuron

# Model definitions for test independence
class SimpleCNN4D(nn.Module):
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
# Helpers
# ===========================================================================

def _all_params_on_device(model, target_device):
    """Check that all parameters and buffers are on target_device.
    Compares device type (not index) since cuda:0 == cuda for practical purposes.
    """
    wrong = []
    for name, p in model.named_parameters():
        if p.device.type != target_device.type:
            wrong.append(f"param:{name} on {p.device}, expected {target_device}")
    for name, b in model.named_buffers():
        if b.device.type != target_device.type:
            wrong.append(f"buffer:{name} on {b.device}, expected {target_device}")
    return wrong


# ===========================================================================
# Tests
# ===========================================================================

class TestDevicePlacement:
    """Verify all model parameters stay on the target device."""

    def test_qcfs_replacement_keeps_device_after_to(self, device):
        """After _replace_activations_cs + .to(device), new QCFS params should be on device."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device)
        _replace_activations_cs(model, (nn.ReLU,), lambda nc: QCFS(L=8, thresh_init=2.0, num_channels=nc))
        # _replace_activations_cs creates new activations on CPU.
        # Caller must move to device after replacement (this is the documented pattern).
        model = model.to(device)
        wrong = _all_params_on_device(model, device)
        assert len(wrong) == 0, f"Parameters on wrong device after .to(device): {wrong}"

    def test_qcfs_replacement_scalar_keeps_device_after_to(self, device):
        """After _replace_activations + .to(device), QCFS params should be on device."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device)
        _replace_activations(model, (nn.ReLU,), lambda: QCFS(L=8, thresh_init=3.0, num_channels=None))
        model = model.to(device)
        wrong = _all_params_on_device(model, device)
        assert len(wrong) == 0, f"Parameters on wrong device after .to(device): {wrong}"

    def test_if_transfer_keeps_device_after_to(self, device):
        """After _transfer_qcfs_to_if + .to(device), IF buffers should be on device."""
        model = SimpleCNN4D(act_factory=lambda: QCFS(L=8, thresh_init=2.0, num_channels=8)).to(device)
        thresholds = [[2.0] * 8]
        _transfer_qcfs_to_if(model, thresholds, channel_wise=True)
        # _transfer_qcfs_to_if creates new IF neurons with buffers on CPU.
        # Caller must move to device after transfer.
        model = model.to(device)
        wrong = _all_params_on_device(model, device)
        assert len(wrong) == 0, f"Parameters/buffers on wrong device after .to(device): {wrong}"

    def test_convert_full_pipeline_device(self, device):
        """After full convert(), all params should be on the target device."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device)

        x = torch.randn(32, 1, 32, 32, device=device)
        y = torch.zeros(32, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)

        snn, _ = convert(
            model,
            loader,
            qcfs_epochs=1,
            if_epochs=1,
            strategy="qcfs_if_ft",
            channel_wise=True,
            device=device,
            verbose=False,
        )

        wrong = _all_params_on_device(snn, device)
        assert len(wrong) == 0, f"Parameters/buffers on wrong device after convert(): {wrong}"

    def test_convert_5d_full_pipeline_device(self, device):
        """After full convert() on 5D model, all params should be on device."""
        model = SimpleCNN5D(act_factory=nn.ReLU).to(device)

        x = torch.randn(16, 4, 1, 32, 32, device=device)
        y = torch.zeros(16, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)

        snn, _ = convert(
            model,
            loader,
            qcfs_epochs=1,
            if_epochs=1,
            strategy="qcfs_if_ft",
            channel_wise=True,
            device=device,
            verbose=False,
        )

        wrong = _all_params_on_device(snn, device)
        assert len(wrong) == 0, f"Parameters/buffers on wrong device after 5D convert(): {wrong}"


class TestDeviceMovement:
    """Verify models can move between devices."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_moves_gpu_to_cpu(self):
        """Model should move from CUDA to CPU without losing parameters."""
        device_cuda = torch.device("cuda")
        device_cpu = torch.device("cpu")

        model = SimpleCNN4D(act_factory=lambda: IFNeuron(thresh=2.0, num_channels=8)).to(device_cuda)
        # Verify on CUDA
        wrong = _all_params_on_device(model, device_cuda)
        assert len(wrong) == 0

        # Move to CPU
        model = model.to(device_cpu)
        wrong = _all_params_on_device(model, device_cpu)
        assert len(wrong) == 0

    def test_if_buffer_moves_with_model(self, device):
        """IF threshold buffers should move with .to(device)."""
        if_cpu = IFNeuron(thresh=2.0, alpha=2.0, num_channels=8)
        if_device = if_cpu.to(device)
        assert if_device.thresh.device.type == device.type, \
            f"IF threshold on {if_device.thresh.device}, expected {device}"

    def test_qcfs_param_moves_with_model(self, device):
        """QCFS threshold parameters should move with .to(device)."""
        qcfs_cpu = QCFS(L=8, thresh_init=2.0, num_channels=8)
        qcfs_device = qcfs_cpu.to(device)
        assert qcfs_device.thresh.device.type == device.type, \
            f"QCFS threshold on {qcfs_device.thresh.device}, expected {device}"


class TestInputDeviceMismatch:
    """Graceful handling of device mismatches."""

    def test_if_handles_input_on_correct_device(self, device):
        """IF neuron should process input on its own device."""
        if_neuron = IFNeuron(thresh=2.0, alpha=2.0, num_channels=8).to(device).eval()
        x = torch.randn(4, 8, 16, 16, device=device)
        out = if_neuron(x)
        assert out.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_cpu_fails_on_cuda_input(self):
        """CPU model should raise on CUDA input (expected PyTorch behavior)."""
        model = SimpleCNN4D(act_factory=lambda: IFNeuron(thresh=2.0)).cpu().eval()
        x = torch.randn(4, 1, 32, 32, device="cuda")
        with pytest.raises(RuntimeError):
            model(x)
