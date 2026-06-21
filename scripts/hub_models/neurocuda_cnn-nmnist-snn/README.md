---
license: mit
tags:
- neurocuda
- spiking-neural-network
- snn
- neuromorphic
- vision
- vision
- event-camera
- n-mnist
- classification
- robotics
- flagship
pipeline_tag: event-camera-vision
---

# neurocuda/cnn-nmnist-snn ✅

Event-camera object classification SNN. Beats the original ANN by 0.18%. 92% sparse — only 8% of neurons fire. Best-in-class conversion. Our flagship model.

## Model Details

- **Task:** event-camera-vision
- **Dataset:** N-MNIST
- **Architecture:** 3-layer CNN (2 input channels, 34×34)
- **Training:** ANN → CS-QCFS → IF + BPTT FT (conversion)
- **Status:** production

## Performance

- **SNN Accuracy:** 99.88% ± 0.02% | **Gap:** -0.18% (BETTER than ANN)
- **Sparsity:** 91.7% ± 0.5%
- **Parameters:** 147,466 (576 KB)
- **Timesteps:** T=16

## Usage

```python
import neurocuda as nc

# Load the pre-converted spiking model
snn, info = nc.hub.load("neurocuda/cnn-nmnist-snn")

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

ANN → CS-QCFS → IF + BPTT FT (conversion)

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
