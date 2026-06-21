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
- conversion
pipeline_tag: reinforcement-learning
---

# neurocuda/dqn-cartpole-snn ⚠️

DQN policy network converted to LIF SNN for CartPole control. Best seed reaches 100% solved. ~29% seed success rate — DQN training produces policies with varying robustness to ReLU→LIF transfer. Use direct LIF training for 100% reliability.

## Model Details

- **Task:** reinforcement-learning
- **Dataset:** CartPole-v1
- **Architecture:** 3-layer MLP DQN (4→128→128→2) with LIF neurons
- **Training:** ANN weight transfer + BPTT FT (conversion, surrogate gradient)
- **Status:** beta

## Performance

- **Best Solved:** 100%
- **Mean Solved:** 19.0% ± 26.0%
- **Sparsity:** 74.5% ± 2.1%
- **Parameters:** 17,922 (70 KB)
- **Timesteps:** T=16

## Usage

```python
import neurocuda as nc

# Load the pre-converted spiking model
snn, info = nc.hub.load("neurocuda/dqn-cartpole-snn")

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

ANN weight transfer + BPTT FT (conversion, surrogate gradient)

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

⚠️ Stochastic benchmark. See neurocuda/lif-dqn-cartpole-snn for the reliable version.
