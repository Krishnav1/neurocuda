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
- resnet
- classification
- deep
- residual
pipeline_tag: image-classification
---

# neurocuda/resnet18-cifar10-snn ✅

Deep residual SNN for CIFAR-10 classification. Gap of 0.95% at T=32. Uses direct QCFS→IF replacement without fine-tuning (standard for deep ResNets). Skip connections handled by Kahn's topology sort in NIR executor.

## Model Details

- **Task:** image-classification
- **Dataset:** CIFAR-10
- **Architecture:** ResNet-18 (CIFAR variant, 8 residual blocks, skip connections)
- **Training:** ANN → QCFS → IF (direct conversion, no fine-tune). T=32.
- **Status:** production

## Performance

- **SNN Accuracy:** 94.61% ± 0.14% | **Gap:** +0.95% (within ANN)
- **Sparsity:** 93.7%
- **Parameters:** 11,173,962 (42.6 MB)
- **Timesteps:** T=32

## Usage

```python
import neurocuda as nc

# Load the pre-converted spiking model
snn, info = nc.hub.load("neurocuda/resnet18-cifar10-snn")

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

- **Validated on:** GPU, CPU, Loihi 2 simulator
- **NIR Export:** Yes — deployable to Loihi 2, SpiNNaker, FPGA

## Conversion Method

ANN → QCFS → IF (direct conversion, no fine-tune). T=32.

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

This is a converted spiking neural network. Accuracy was measured on the full test set with ≥3 seeds (where noted). Performance may vary on different hardware backends. See the NeuroCUDA README for detailed benchmarking methodology.
