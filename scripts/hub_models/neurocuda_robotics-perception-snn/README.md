---
license: mit
tags:
- neurocuda
- spiking-neural-network
- snn
- neuromorphic
- vision
- vision
- robotics
- event-camera
- n-mnist
- perception
- energy
pipeline_tag: robotics-perception
---

# neurocuda/robotics-perception-snn ✅

Full robotics perception pipeline — event camera → SNN → deploy. 99.95% accuracy (beats ANN). 92% sparse. 49% energy reduction vs ANN. NIR-exported and ready for Loihi 2 / SpiNNaker / FPGA deployment.

## Model Details

- **Task:** robotics-perception
- **Dataset:** N-MNIST (event camera)
- **Architecture:** 3-layer CNN (2 input channels, 34×34), 5D-native
- **Training:** ANN → CS-QCFS → IF + BPTT FT (conversion, 5 epochs, 20K data)
- **Status:** production

## Performance

- **SNN Accuracy:** 99.95% | **Gap:** -0.25% (BETTER than ANN)
- **Sparsity:** 92.06%
- **Energy/Inference:** 13.02 µJ
- **Energy vs ANN:** 49% reduction
- **Parameters:** 147,466 (576 KB)
- **Timesteps:** T=16

## Usage

```python
import neurocuda as nc

# Load the pre-converted spiking model
snn, info = nc.hub.load("neurocuda/robotics-perception-snn")

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

- **Validated on:** GPU, CPU, Loihi 2 simulator, NIR-exported
- **NIR Export:** Yes — deployable to Loihi 2, SpiNNaker, FPGA

## Conversion Method

ANN → CS-QCFS → IF + BPTT FT (conversion, 5 epochs, 20K data)

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
