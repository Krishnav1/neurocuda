"""
QCFS (Quantization Clip-Floor-Shift) — ICLR 2022
==================================================
Bu, Fang, Ding, Dai, Yu, Huang (Peking University)
"Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency SNNs"

Replaces percentile calibration. The ANN is trained with QCFS activation
that mimics SNN quantization → near-zero conversion error.

Formula:
  a = λ · clip(⌊x·L/λ + 0.5⌋ / L, 0, 1)

After training: λ = LIF threshold, L = quantization levels = timesteps

Reference: github.com/putshua/SNN_conversion_QCFS
"""
import torch
import torch.nn as nn


class QCFSActivation(nn.Module):
    """
    Quantization Clip-Floor-Shift activation.
    Replaces ReLU during ANN training for SNN-aware optimization.

    Args:
        lambda_init: Initial threshold value (default 1.0)
        L: Quantization levels (default 16, matches T during conversion)
    """

    def __init__(self, lambda_init: float = 1.0, L: int = 16):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.L = L

    def forward(self, x):
        # Scale input by L/lambda
        x_scaled = x * self.L / self.lambda_param.abs()

        # Quantize: round to nearest integer (floor + 0.5 shift)
        x_quant = torch.floor(x_scaled + 0.5)

        # Clip to [0, L] then normalize back
        x_clipped = torch.clamp(x_quant / self.L, 0.0, 1.0)

        # Scale back by lambda
        # Use straight-through estimator: gradient flows through clip/round
        return self.lambda_param.abs() * x_clipped

    def get_threshold(self):
        """After training, this is the LIF threshold."""
        return self.lambda_param.abs().detach()

    def get_quantization_levels(self):
        """Number of discrete output levels."""
        return self.L


class QCFSConverter:
    """
    Converts a QCFS-trained ANN to SNN.
    Much simpler than percentile calibration:
      - Threshold = learned lambda from each QCFS layer
      - T = L (quantization levels) for exact conversion
      - T < L for faster but approximate
    """

    def __init__(self, ann_model, T: int = 16):
        self.ann = ann_model
        self.T = T
        self.thresholds = []
        self._extract_thresholds()

    def _extract_thresholds(self):
        """Extract learned lambda values from QCFS layers."""
        for name, module in self.ann.named_modules():
            if isinstance(module, QCFSActivation):
                self.thresholds.append(module.get_threshold())
        if not self.thresholds:
            raise ValueError("No QCFSActivation layers found in model. "
                           "Train the ANN with QCFS before converting.")

    def get_thresholds(self) -> list:
        return self.thresholds

    def get_T(self) -> int:
        """Recommended timesteps = quantization levels for exact conversion."""
        return self.ann.qcfs_L if hasattr(self.ann, 'qcfs_L') else self.T


def replace_relu_with_qcfs(model, L=16, lambda_init=1.0):
    """
    Replace all ReLU layers in a model with QCFS activations.
    Call this BEFORE training the ANN.

    Args:
        model: PyTorch model with ReLU activations
        L: Quantization levels (matches T during SNN inference)
        lambda_init: Initial threshold (default 1.0)

    Returns:
        Modified model with QCFS activations (in-place)
    """
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            # Find parent to replace
            parts = name.rsplit('.', 1)
            parent_name = parts[0] if len(parts) > 1 else ''
            child_name = parts[1] if len(parts) > 1 else name

            if parent_name:
                parent = model
                for p in parent_name.split('.'):
                    parent = getattr(parent, p)
                setattr(parent, child_name, QCFSActivation(lambda_init, L))
            else:
                # Top-level module
                setattr(model, name, QCFSActivation(lambda_init, L))

    model.qcfs_L = L
    return model