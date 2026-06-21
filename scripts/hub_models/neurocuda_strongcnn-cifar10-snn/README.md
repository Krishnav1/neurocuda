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
- strongcnn
- classification
- medium
pipeline_tag: image-classification
---

# neurocuda/strongcnn-cifar10-snn ⚠️

7-layer CNN SNN for CIFAR-10. Gap of 6% with conversion + fine-tuning. Larger gap than ResNet-18 due to higher non-linearity. Good for studying conversion challenges in medium-depth networks.

## Model Details

- **Task:** image-classification
- **Dataset:** CIFAR-10
- **Architecture:** StrongCNN (7-layer, BatchNorm, wider channels)
- **Training:** ANN → QCFS → IF + BPTT FT (conversion + fine-tune)
- **Status:** beta

## Performance

- **SNN Accuracy:** 74.3% | **Gap:** +6.00% (within ANN)
- **Parameters:** 4,800,000 (18.3 MB)
- **Timesteps:** T=16

## Usage

```python
import neurocuda as nc

# Load the pre-converted spiking model
snn, info = nc.hub.load("neurocuda/strongcnn-cifar10-snn")

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

ANN → QCFS → IF + BPTT FT (conversion + fine-tune)

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

⚠️ Single seed result. Multi-seed verification pending.
