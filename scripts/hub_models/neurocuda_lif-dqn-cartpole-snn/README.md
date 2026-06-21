---
license: mit
tags:
- neurocuda
- spiking-neural-network
- snn
- neuromorphic
- control
- control
- reinforcement-learning
- cartpole
- dqn
- rl
- direct-training
pipeline_tag: reinforcement-learning
---

# neurocuda/lif-dqn-cartpole-snn ✅

LIF SNN DQN trained from scratch via BPTT with surrogate gradients. 100% reliable — always reaches 100% solved. The recommended approach for spiking reinforcement learning. No conversion needed — native SNN training.

## Model Details

- **Task:** reinforcement-learning
- **Dataset:** CartPole-v1
- **Architecture:** 3-layer MLP DQN (4→128→128→2) with LIF neurons
- **Training:** Direct LIF SNN training from scratch (BPTT + surrogate gradient)
- **Status:** production

## Performance

- **Solved:** 100% (solved at ~350 episodes)
- **Sparsity:** 68.5%
- **Parameters:** 17,922 (70 KB)
- **Timesteps:** T=16

## Usage

```python
import neurocuda as nc

# Load the pre-converted spiking model
snn, info = nc.hub.load("neurocuda/lif-dqn-cartpole-snn")

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
- **NIR Export:** Not yet

## Conversion Method

Direct LIF SNN training from scratch (BPTT + surrogate gradient)

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
