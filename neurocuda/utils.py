"""Utility functions for NeuroCUDA."""
import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate


def energy_estimate(model, input_shape=(1, 3, 32, 32), T=64, spike_rate=0.20):
    """
    Theoretical energy comparison: ANN (GPU) vs SNN (Neuromorphic).

    ANN: Every parameter used every inference (~50 pJ/FLOP on GPU)
    SNN: Only spiking neurons consume energy (~0.1 pJ/spike on Loihi)

    Returns dict with energy estimates and savings ratio.
    """
    total_params = sum(p.numel() for p in model.parameters())
    ann_flops = total_params * 2  # multiply-add = 2 ops
    snn_spike_ops = total_params * spike_rate * T

    # Energy constants (picojoules)
    GPU_PJ_PER_FLOP = 50
    NEURO_PJ_PER_SPIKE = 0.1

    gpu_energy_uj = (ann_flops * GPU_PJ_PER_FLOP) / 1e6  # microjoules
    neuro_energy_uj = (snn_spike_ops * NEURO_PJ_PER_SPIKE) / 1e6
    ratio = gpu_energy_uj / max(neuro_energy_uj, 1e-6)

    return {
        "ann_flops": ann_flops,
        "snn_spike_ops": snn_spike_ops,
        "gpu_energy_uj": gpu_energy_uj,
        "neuro_energy_uj": neuro_energy_uj,
        "energy_ratio": ratio,
    }


def fold_batchnorm(model):
    """Fold BatchNorm into preceding Conv/Linear. Set BN to identity."""
    # Find Conv→BN pairs by examining module names
    pairs = []
    prev = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prev = name
        elif isinstance(module, nn.BatchNorm2d) and prev is not None:
            pairs.append((prev, name))
            prev = None

    for conv_name, bn_name in pairs:
        conv = dict(model.named_modules())[conv_name]
        bn = dict(model.named_modules())[bn_name]
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(conv.out_channels))
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        conv.weight.data *= s.view(-1, 1, 1, 1)
        conv.bias.data = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)
        bn.weight.data = torch.ones_like(bn.weight)
        bn.bias.data.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0 - bn.eps)

    return model


def validate_snn(snn_model, test_loader, device="cuda"):
    """Evaluate SNN accuracy on test set."""
    snn_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = snn_model(data)
            correct += out.max(1)[1].eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total
