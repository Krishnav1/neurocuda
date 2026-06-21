"""
Unit tests for neuron models: QCFS, IFNeuron, LIFNeuron.

Tests:
  - QCFS: per-layer vs per-channel, threshold shapes, forward pass, learnability
  - IFNeuron: binary spikes, state management, per-channel mode, surrogate gradient
  - LIFNeuron: leaky dynamics, spike patterns, reset behavior
"""

import pytest
import torch
import torch.nn as nn
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import QCFS, IFNeuron, LIFNeuron, reset_spiking, surrogate_spike


# ===========================================================================
# QCFS
# ===========================================================================

class TestQCFS:
    """Quantized-Clip Floor-Shift activation tests."""

    def test_per_layer_scalar_threshold(self):
        """Per-layer QCFS has a scalar threshold."""
        qcfs = QCFS(L=8, thresh_init=3.0, num_channels=None)
        assert qcfs.num_channels is None
        assert qcfs.thresh.ndim == 0  # scalar
        assert qcfs.thresh.item() == pytest.approx(3.0, abs=0.01)

    def test_per_channel_vector_threshold(self):
        """Per-channel QCFS has a vector threshold with correct shape."""
        qcfs = QCFS(L=8, thresh_init=2.0, num_channels=16)
        assert qcfs.num_channels == 16
        assert qcfs.thresh.shape == (16,)
        assert qcfs.thresh[0].item() == pytest.approx(2.0, abs=0.01)

    def test_forward_output_range(self):
        """QCFS output is bounded in [0, lambda]."""
        qcfs = QCFS(L=8, thresh_init=4.0, num_channels=None)
        x = torch.randn(4, 3, 8, 8)  # random input, some negative
        out = qcfs(x)
        thresh = qcfs.thresh.abs() + 1e-4
        assert out.min() >= 0.0, f"Output below 0: {out.min()}"
        assert out.max() <= thresh.item() + 0.1, f"Output above thresh: {out.max()} vs {thresh.item()}"

    def test_per_channel_broadcast(self):
        """Per-channel QCFS broadcasts correctly over spatial dims."""
        qcfs = QCFS(L=8, thresh_init=2.0, num_channels=8)
        x = torch.randn(4, 8, 16, 16)
        out = qcfs(x)
        assert out.shape == (4, 8, 16, 16)

    def test_negative_input_clamped(self):
        """Negative inputs should be clamped to ~0 (floor→0, then clamp)."""
        qcfs = QCFS(L=8, thresh_init=5.0, num_channels=None)
        x = -10.0 * torch.ones(1, 1, 4, 4)
        out = qcfs(x)
        assert out.max() <= 0.1, f"Negative input should produce ~0 output, got {out.max()}"

    def test_threshold_is_learnable(self):
        """QCFS threshold should be a trainable parameter."""
        qcfs = QCFS(L=8, thresh_init=3.0, num_channels=None)
        assert isinstance(qcfs.thresh, nn.Parameter)
        assert qcfs.thresh.requires_grad

    def test_per_channel_threshold_individual_values(self):
        """Per-channel thresholds can have different values."""
        qcfs = QCFS(L=8, thresh_init=2.0, num_channels=4)
        # Manually set different thresholds
        with torch.no_grad():
            qcfs.thresh.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        x = torch.ones(1, 4, 1, 1) * 5.0  # All channels input 5
        out = qcfs(x)
        # Each channel should be clamped to its own threshold
        assert out[0, 0].max() < out[0, 3].max(), "Channel 0 should be clamped lower than channel 3"


# ===========================================================================
# IFNeuron
# ===========================================================================

class TestIFNeuron:
    """Integrate-and-Fire neuron tests."""

    def test_eval_mode_produces_binary_spikes(self):
        """In eval mode, IF output is exactly 0 or threshold (binary)."""
        if_neuron = IFNeuron(thresh=2.0, alpha=2.0).eval()
        x = torch.randn(1, 4, 4, 4)
        out = if_neuron(x)
        thresh_val = if_neuron.thresh.item()
        unique_vals = out.unique()
        for v in unique_vals:
            assert v.item() == pytest.approx(0.0, abs=1e-4) or \
                   v.item() == pytest.approx(thresh_val, abs=1e-4), \
                   f"Output value {v.item()} is neither 0 nor {thresh_val}"

    def test_training_mode_uses_surrogate(self):
        """In training mode, IF passes through surrogate gradient (output still binary)."""
        if_neuron = IFNeuron(thresh=2.0, alpha=2.0).train()
        x = torch.randn(1, 4, 4, 4)
        out = if_neuron(x)
        # Output should still be binary in training (surrogate only affects backward)
        thresh_val = if_neuron.thresh.item()
        unique_vals = out.unique()
        for v in unique_vals:
            assert v.item() == pytest.approx(0.0, abs=1e-4) or \
                   v.item() == pytest.approx(thresh_val, abs=1e-4), \
                   f"Output value {v.item()} is neither 0 nor {thresh_val}"

    def test_state_resets(self):
        """reset() clears membrane potential."""
        if_neuron = IFNeuron(thresh=2.0, alpha=2.0).eval()
        x = torch.ones(1, 4, 4, 4)  # Positive input builds up membrane
        _ = if_neuron(x)
        assert if_neuron.v is not None, "Membrane should be initialized after forward"
        reset_spiking(if_neuron)
        assert if_neuron.v is None, "Membrane should be None after reset"

    def test_per_layer_scalar(self):
        """Per-layer IF accepts scalar threshold."""
        if_neuron = IFNeuron(thresh=3.0, alpha=2.0, num_channels=None)
        assert if_neuron.thresh.ndim == 0
        assert if_neuron.thresh.item() == pytest.approx(3.0)

    def test_per_channel_vector(self):
        """Per-channel IF creates vector threshold."""
        if_neuron = IFNeuron(thresh=1.5, alpha=2.0, num_channels=16)
        assert if_neuron.thresh.shape == (16,)
        assert if_neuron.thresh[0].item() == pytest.approx(1.5)

    def test_per_channel_broadcast(self):
        """Per-channel IF broadcasts correctly."""
        if_neuron = IFNeuron(thresh=2.0, alpha=2.0, num_channels=8).eval()
        x = torch.randn(4, 8, 16, 16)
        out = if_neuron(x)
        assert out.shape == (4, 8, 16, 16)

    def test_threshold_is_not_trainable(self):
        """IF threshold is a buffer, not a parameter (matches converter design)."""
        if_neuron = IFNeuron(thresh=2.0, alpha=2.0)
        assert not if_neuron.thresh.requires_grad

    def test_large_input_spikes(self):
        """Very large input should cause spiking."""
        if_neuron = IFNeuron(thresh=1.0, alpha=2.0).eval()
        x = 10.0 * torch.ones(1, 1, 4, 4)
        out = if_neuron(x)
        assert (out > 0).any(), "Large input should cause spikes"

    def test_zero_input_no_spikes(self):
        """Zero input should not cause spikes (membrane starts at thresh/2)."""
        if_neuron = IFNeuron(thresh=2.0, alpha=2.0).eval()
        x = torch.zeros(1, 4, 4, 4)
        out = if_neuron(x)
        # With zero input, v stays at thresh/2 < thresh → no spikes
        assert (out == 0).all(), "Zero input should not cause spikes"


# ===========================================================================
# LIFNeuron
# ===========================================================================

class TestLIFNeuron:
    """Leaky Integrate-and-Fire neuron tests."""

    def test_leak_reduces_membrane(self):
        """With beta < 1, membrane should decay without input."""
        lif = LIFNeuron(threshold=2.0, beta=0.5, alpha=2.0).eval()
        # First step: strong input builds membrane
        x1 = 4.0 * torch.ones(1, 4, 4, 4)
        _ = lif(x1)
        v_after_first = lif.v.clone()
        # Second step: zero input → leak reduces v
        x2 = torch.zeros(1, 4, 4, 4)
        _ = lif(x2)
        v_after_leak = lif.v.clone()
        assert (v_after_leak < v_after_first).all(), \
            "LIF membrane should decay with beta < 1"

    def test_perfect_memory_beta_1(self):
        """With beta=1, membrane does not decay."""
        lif = LIFNeuron(threshold=10.0, beta=1.0, alpha=2.0).eval()
        x1 = 3.0 * torch.ones(1, 1, 2, 2)
        _ = lif(x1)
        v1 = lif.v.clone()
        _ = lif(torch.zeros(1, 1, 2, 2))
        # With beta=1 and no spike, v should be the same
        assert torch.allclose(lif.v, v1, atol=1e-5), \
            "beta=1 should preserve membrane exactly"

    def test_spike_resets_membrane(self):
        """After spike, membrane should be reduced by threshold (soft reset)."""
        lif = LIFNeuron(threshold=1.0, beta=1.0, alpha=2.0).eval()
        # Input just above threshold: 1.5 → v=0+1.5=1.5 → spike → v=1.5-1.0=0.5 < threshold
        x = 1.5 * torch.ones(1, 1, 4, 4)
        _ = lif(x)
        assert (lif.v < lif.threshold).all(), \
            f"Membrane should be below threshold after spike + soft reset, got max={lif.v.max().item()}"

    def test_no_spike_threshold_untrainable(self):
        """LIF threshold should not be trainable (matches design)."""
        lif = LIFNeuron(threshold=1.0, beta=0.5, alpha=2.0)
        assert not lif.threshold.requires_grad

    def test_reset_clears_state(self):
        """reset() clears LIF membrane."""
        lif = LIFNeuron(threshold=2.0, beta=0.5).eval()
        _ = lif(torch.ones(1, 4, 4, 4))
        assert lif.v is not None
        lif.reset()
        assert lif.v is None

    def test_eval_mode_hard_spikes(self):
        """In eval mode, LIF produces binary spikes (0 or 1)."""
        lif = LIFNeuron(threshold=2.0, beta=0.5, alpha=2.0).eval()
        x = torch.randn(1, 4, 4, 4)
        out = lif(x)
        unique_vals = out.unique()
        # LIF outputs 0 or 1 (unlike IF which outputs 0 or threshold)
        for v in unique_vals:
            assert v.item() in [0.0, 1.0], \
                f"LIF spike should be 0 or 1, got {v.item()}"
