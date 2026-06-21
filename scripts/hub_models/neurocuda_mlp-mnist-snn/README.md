---
license: mit
tags:
- neurocuda
- spiking-neural-network
- snn
- neuromorphic
- vision
- vision
- digit
- mnist
- classification
- quickstart
- beginner
pipeline_tag: digit-classification
---

# neurocuda/mlp-mnist-snn ✅

Simple MLP SNN for MNIST digit classification. The classic beginner example — great for learning the conversion pipeline. Converts in under 1 minute on CPU.

## Model Details

- **Task:** digit-classification
- **Dataset:** MNIST
- **Architecture:** 3-layer MLP (784→256→256→10)
- **Training:** ANN → QCFS → IF (direct conversion, no FT)
- **Status:** production

## Performance

- **SNN Accuracy:** 97.4% | **Gap:** +0.40% (within ANN)
- **Parameters:** 269,322 (1052 KB)
- **Timesteps:** T=16

## Usage

```python
import neurocuda as nc

# Load the pre-converted spiking model
snn, info = nc.hub.load("neurocuda/mlp-mnist-snn")

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

ANN → QCFS → IF (direct conversion, no FT)

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
