"""
models.py — shared definitions for NeuroCUDA Gates 2 & 3.

Contains:
  - GradFloor: straight-through estimator for floor() so gradients reach lambda.
  - QCFS: Quantization Clip-Floor-Shift activation with a LEARNABLE per-layer
          threshold (lambda). This is the activation used to fine-tune the ANN
          before conversion. (Bu et al., 2022, "Optimal ANN-SNN Conversion".)
  - IFNeuron: stateful integrate-and-fire neuron for SNN inference (soft reset).
  - resnet18_cifar(act_layer): CIFAR-variant ResNet-18 with a swappable activation.
  - build_snn_from_qcfs(model): clones a QCFS-trained model, replacing each QCFS
          with an IFNeuron whose threshold = the learned lambda.

Design note: every activation site is a single swappable module, so the SAME
architecture is used for ANN (ReLU), conversion-tuning (QCFS), and SNN (IF).
Conv/BN/FC weights are identical across all three; only activations change.
"""

import copy
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Straight-through floor: forward = floor(x), backward = identity.
# Without this, the floor() in QCFS blocks gradients and lambda never moves
# (this was the frozen-lambda bug: lambda stuck at ~1.0 for early layers).
# ---------------------------------------------------------------------------
class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


myfloor = GradFloor.apply


# ---------------------------------------------------------------------------
# QCFS activation. L = quantization steps. thresh (lambda) is learnable.
#   a = lambda * clip( floor( z/lambda * L + 0.5 ) / L , 0, 1 )
#
# Supports both per-layer (scalar λ) and per-channel (vector λ) modes.
# Set num_channels=None for per-layer, num_channels=C for per-channel.
# ---------------------------------------------------------------------------
class QCFS(nn.Module):
    def __init__(self, L=8, thresh_init=4.0, num_channels=None):
        super().__init__()
        self.num_channels = num_channels
        if num_channels is not None:
            # Per-channel: one learnable threshold per channel
            self.thresh = nn.Parameter(
                torch.ones(num_channels) * float(thresh_init))
        else:
            # Per-layer: single learnable scalar
            self.thresh = nn.Parameter(torch.tensor(float(thresh_init)))
        self.L = L

    def forward(self, x):
        # Guard against non-positive thresholds during training.
        thresh = self.thresh.abs() + 1e-4
        # Reshape for broadcasting if per-channel
        if self.num_channels is not None and thresh.dim() == 1:
            # x: (B, C, H, W) or (B*T, C, H, W) — broadcast over C
            thresh = thresh.view(1, -1, *([1] * (x.dim() - 2)))
        x = x / thresh
        x = torch.clamp(x, 0.0, 1.0)
        x = myfloor(x * self.L + 0.5) / self.L   # STE floor -> gradient reaches thresh
        x = x * thresh
        return x


# ---------------------------------------------------------------------------
# IF neuron for SNN inference. Threshold is a fixed scalar (the learned lambda).
# Output is a GRADED spike (spike * thresh) so the inter-layer scaling matches
# the QCFS activation range [0, lambda]. Initial membrane = thresh/2 (shift=0.5).
# ---------------------------------------------------------------------------
class IFNeuron(nn.Module):
    """Integrate-and-Fire neuron for SNN conversion + inference.

    Forward: binary spike (0 or threshold) with soft reset.
    Training mode: uses surrogate gradient (atan) for backward pass,
                   enabling post-conversion fine-tuning via BPTT.
    Eval mode: hard threshold (standard IF inference).

    Supports both per-layer (scalar thresh) and per-channel (vector thresh).
    Set num_channels=None for per-layer, num_channels=C for per-channel.
    """

    def __init__(self, thresh=1.0, alpha=2.0, num_channels=None):
        super().__init__()
        self.num_channels = num_channels
        self.alpha = float(alpha)
        if num_channels is not None:
            self.register_buffer(
                'thresh', torch.ones(num_channels) * float(thresh))
        else:
            self.register_buffer('thresh', torch.tensor(float(thresh)))
        self.v = None

    def reset(self):
        self.v = None

    def forward(self, x):
        # Reshape threshold for broadcasting if per-channel
        thresh = self.thresh
        if self.num_channels is not None and thresh.dim() == 1:
            thresh = thresh.view(1, -1, *([1] * (x.dim() - 2)))

        if self.v is None:
            self.v = torch.ones_like(x) * (thresh * 0.5)
        self.v = self.v + x

        if self.training:
            # Surrogate gradient enables BPTT through the spike
            spike = surrogate_spike(self.v, thresh, self.alpha)
        else:
            spike = (self.v >= thresh).float()

        self.v = self.v - spike * thresh        # soft reset (subtractive)
        return spike * thresh


# ---------------------------------------------------------------------------
# Surrogate gradient: arctan approximation.
# Forward = hard step (v >= threshold). Backward = smooth atan derivative.
# Used by LIFNeuron for direct SNN training via BPTT.
# ---------------------------------------------------------------------------
class _SurrogateGradient(torch.autograd.Function):
    """Straight-through estimator with atan surrogate gradient.

    Forward:  s = 1 if v >= threshold else 0   (hard threshold, binary spike)
    Backward: ds/dv ≈ alpha / (2 * (1 + (π/2 · alpha · (v - threshold))²))

    α (alpha) = 2.0 gives a smooth gradient that works well in practice.
    """
    @staticmethod
    def forward(ctx, v, threshold, alpha=2.0):
        thresh_t = threshold.detach().clone() if isinstance(threshold, torch.Tensor) else torch.tensor(float(threshold), device=v.device, dtype=v.dtype)
        ctx.save_for_backward(v, thresh_t)
        ctx.alpha = alpha
        return (v >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        v, threshold = ctx.saved_tensors
        alpha = ctx.alpha
        # atan surrogate gradient: smooth around threshold, ~0 far away
        inner = (torch.pi / 2.0) * alpha * (v - threshold)
        grad_v = alpha / (2.0 * (1.0 + inner * inner))
        return grad_output * grad_v, None, None  # no grad for threshold, alpha


def surrogate_spike(v, threshold, alpha=2.0):
    """Binary spike with surrogate gradient for training."""
    return _SurrogateGradient.apply(v, threshold, alpha)


# ---------------------------------------------------------------------------
# LIF neuron for DIRECT SNN TRAINING (surrogate gradient BPTT).
# Dynamics:  v[t] = beta * v[t-1] + input[t]
#            spike[t] = 1 if v[t] >= threshold else 0   (surrogate for backward)
#            v[t] = v[t] - spike[t] * threshold          (soft reset)
#
# The spike is BINARY (0 or 1) — a REAL spiking neuron.
# During inference (eval mode), the surrogate is skipped and spikes are hard.
# ---------------------------------------------------------------------------
class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron trainable via surrogate gradient BPTT.

    Args:
        threshold: firing threshold (scalar)
        beta:     leak factor. 0 = no memory (stateless), 1 = perfect memory.
                  Typical: 0.5 for temporal smoothing.
        alpha:    surrogate gradient sharpness. Higher = steeper gradient.
                  Typical: 2.0 (default in snnTorch).
    """

    def __init__(self, threshold=1.0, beta=0.5, alpha=2.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)), requires_grad=False)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.v = None  # membrane potential (stateful across timesteps)

    def reset(self):
        self.v = None

    def forward(self, x):
        if self.v is None:
            self.v = torch.zeros_like(x)

        # Leak + integrate
        self.v = self.beta * self.v + x

        # Spike with surrogate gradient (train) or hard step (eval)
        if self.training:
            spike = surrogate_spike(self.v, self.threshold, self.alpha)
        else:
            spike = (self.v >= self.threshold).float()

        # Soft reset (subtract threshold for each spike)
        self.v = self.v - spike * self.threshold

        return spike  # BINARY spike — real spiking neuron


# ---------------------------------------------------------------------------
# Utility to reset all spiking neurons in a model.
# ---------------------------------------------------------------------------
def reset_spiking(model):
    """Reset membrane potential of all IF/LIF neurons in a model."""
    for m in model.modules():
        if isinstance(m, (IFNeuron, LIFNeuron)):
            m.reset()


# ---------------------------------------------------------------------------
# CIFAR ResNet-18 with a swappable activation factory `act_layer()`.
# CIFAR variant: 3x3 stem, stride 1, no initial maxpool.
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, act_layer):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = act_layer()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet18CIFAR(nn.Module):
    def __init__(self, act_layer, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = act_layer()
        self.layer1 = self._make_layer(64, 2, 1, act_layer)
        self.layer2 = self._make_layer(128, 2, 2, act_layer)
        self.layer3 = self._make_layer(256, 2, 2, act_layer)
        self.layer4 = self._make_layer(512, 2, 2, act_layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n_blocks, stride, act_layer):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, act_layer))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet18_cifar(act_layer, num_classes=10):
    return ResNet18CIFAR(act_layer, num_classes)


# ---------------------------------------------------------------------------
# Build an SNN (IF neurons) from a QCFS-trained model by copying its structure
# and weights, and replacing each QCFS with an IFNeuron carrying its lambda.
# ---------------------------------------------------------------------------
def build_snn_from_qcfs(qcfs_model):
    """Build an SNN (IF neurons) from a QCFS-trained model.

    Handles both per-layer (scalar λ) and per-channel (vector λ) QCFS thresholds.
    """
    snn = copy.deepcopy(qcfs_model)

    def replace(module):
        for name, child in module.named_children():
            if isinstance(child, QCFS):
                nc = child.num_channels
                thresh = child.thresh.abs() + 1e-4
                if nc is not None:
                    # Per-channel: create IF with matching channel count
                    new_if = IFNeuron(thresh=1.0, alpha=2.0, num_channels=nc)
                    new_if.thresh.copy_(thresh.data)
                else:
                    # Per-layer: scalar threshold
                    new_if = IFNeuron(
                        thresh=thresh.item() if thresh.numel() == 1
                        else thresh.mean().item(), alpha=2.0)
                setattr(module, name, new_if)
            else:
                replace(child)

    replace(snn)
    return snn


def reset_snn(snn):
    for m in snn.modules():
        if isinstance(m, IFNeuron):
            m.reset()
