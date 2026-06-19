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
# ---------------------------------------------------------------------------
class QCFS(nn.Module):
    def __init__(self, L=8, thresh_init=4.0):
        super().__init__()
        # lambda as a learnable scalar parameter, initialised positive.
        self.thresh = nn.Parameter(torch.tensor(float(thresh_init)))
        self.L = L

    def forward(self, x):
        # Guard against a non-positive threshold during training.
        thresh = self.thresh.abs() + 1e-4
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
    def __init__(self, thresh=1.0):
        super().__init__()
        self.thresh = float(thresh)
        self.v = None

    def reset(self):
        self.v = None

    def forward(self, x):
        if self.v is None:
            self.v = torch.ones_like(x) * (self.thresh * 0.5)
        self.v = self.v + x
        spike = (self.v >= self.thresh).float()
        self.v = self.v - spike * self.thresh        # soft reset (subtractive)
        return spike * self.thresh


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
    snn = copy.deepcopy(qcfs_model)

    def replace(module):
        for name, child in module.named_children():
            if isinstance(child, QCFS):
                setattr(module, name, IFNeuron(thresh=child.thresh.abs().item() + 1e-4))
            else:
                replace(child)

    replace(snn)
    return snn


def reset_snn(snn):
    for m in snn.modules():
        if isinstance(m, IFNeuron):
            m.reset()
