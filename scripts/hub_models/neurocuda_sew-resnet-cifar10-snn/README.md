---
license: mit
tags:
- neurocuda
- spiking-neural-network
- snn
- neuromorphic
- vision
- vision
- cifar-10
- sew-resnet
- classification
- direct-training
- deep
pipeline_tag: image-classification
---

# neurocuda/sew-resnet-cifar10-snn ⚠️

SEW-ResNet trained directly as an SNN from scratch (no ANN→SNN conversion). 67.7% at T=8 timesteps. Demonstrates direct SNN training with surrogate gradients. Accuracy improves with more timesteps and training epochs.

## Model Details

- **Task:** image-classification
- **Dataset:** CIFAR-10
- **Architecture:** SEW-ResNet (Spiking Element-Wise ResNet, 18 layers)
- **Training:** Direct SNN training from scratch (BPTT, surrogate gradient, 50 epochs)
- **Status:** beta

## Performance

- **SNN Accuracy:** 67.7%
- **Parameters:** 11,170,000 (42.6 MB)
- **Timesteps:** T=8

## Usage

```python
import neurocuda as nc

# Load the pre-converted spiking model
snn, info = nc.hub.load("neurocuda/sew-resnet-cifar10-snn")

# The model is already spiking — binary IF/LIF spikes, stateful membrane
snn.eval()

# 4D input (single frame)
import torch
x = torch.randn(1, 2, 34, 34)  # Adjust channels/size for your model
output = snn(x)

# 5D input (temporal — event cameras, video)
x5 = torch.randn(2, 16, 2, 34, 34)  # (Batch, Timesteps, Channels, H, W)
output5 = snn(x5)
```

## Hardware Compatibility

- **Validated on:** GPU, CPU
- **NIR Export:** Yes — deployable to Loihi 2, SpiNNaker, FPGA

## Conversion Method

Direct SNN training from scratch (BPTT, surrogate gradient, 50 epochs)

## Citation

```bibtex
@software{neurocuda2026,
  title    = {NeuroCUDA: A PyTorch-to-Neuromorphic Compiler},
  author   = {Krishna Varma},
  year     = {2026},
  url      = {https://github.com/neurocuda/neurocuda}
}
```

## Limitations

⚠️ Single seed, 50 epochs. Extended training (200+ epochs) expected to reach 85%+.
