# Lava / Loihi 2 Setup for NeuroCUDA

## Overview

NeuroCUDA deploys to Loihi 2 via:

```
nc.convert() → nc.to_nir() / nir_bridge → Lava import_from_nir → Loihi2SimCfg / Loihi2HwCfg
```

API:

```python
import neurocuda as nc

result = nc.compile(snn, target="loihi2_lava", T=32)
out = result["backend"].run(result["compiled_model"], input_data)

report = nc.verify(snn, test_loader, backends=["gpu", "cpu", "loihi", "loihi2_lava"])
```

## Environment Requirements

| Component | Requirement |
|-----------|-------------|
| **nir** | `pip install nir nirtorch` (Python 3.10+) |
| **lava-nc** | Python **3.10** (not 3.11+ on PyPI as of June 2026) |
| **Loihi2SimCfg** | Public Lava install |
| **Loihi2HwCfg** | INRC membership + proprietary Loihi Magma extension |
| **nir_to_lava** | Bundled with NIR repo / INRC toolchain |

### Development machine (no Lava)

If `lava-nc` cannot install (e.g. Python 3.12 on Windows):

- `loihi2_lava` backend still **exports official NIR** and runs **NeuroCUDA Loihi quant sim**
- `metadata["execution_mode"]` = `neurocuda_loihi_sim` (honest label)
- GATE L2 validation works via `nc.verify()` comparing gpu/cpu/loihi/loihi2_lava

### INRC Linux (physical silicon)

1. Join [Intel Neuromorphic Research Community](https://www.intel.com/content/www/us/en/research/neuromorphic-community.html)
2. Install Lava + Loihi extension on **Python 3.10** Ubuntu
3. `export INRC_LOIHI=1`
4. Run: `python scripts/run_loihi_hw_benchmark.py --hub mlp-mnist-snn`

## Validation Gates

| Gate | Command | Pass criteria |
|------|---------|---------------|
| **L0** | `python -c "from neurocuda.backends.nir_bridge import snn_to_nir_graph"` | NIR builds |
| **L1** | `pytest tests/test_lava_integration.py::test_gate_l1_single_if_neuron` | 0 spike diffs |
| **L2** | `python reproduce.py --lava-gate` | MLP MNIST gap ≤2%, acc ≥95.4% |
| **L4** | `python scripts/run_loihi_hw_benchmark.py` | Physical Loihi ≥95.4% |

## Known Issues (from NIR / Lava community)

1. **Parameter shift on fixed_pt** ([NIR #111](https://github.com/neuromorphs/NIR/issues/111))  
   Run `python scripts/audit_nir_lava_params.py --hub mlp-mnist-snn`

2. **Accuracy drop SLAYER→Lava** ([Lava #891](https://github.com/lava-nc/lava/discussions/891))  
   Match input encoding, T, reset, and readout (voltage sum vs spikes)

3. **reset_interval** must be power of 2 on hardware (use T=32 or T=64)

4. **Public Lava repos archived** — Intel shipping new SDK for Loihi 3; INRC provides current Loihi 2 access

## Files

| File | Purpose |
|------|---------|
| `neurocuda/backends/loihi2_lava.py` | Lava backend |
| `neurocuda/backends/nir_bridge.py` | PyTorch → official NIR |
| `neurocuda/verify.py` | `nc.verify()` cross-backend CI |
| `scripts/audit_nir_lava_params.py` | Parameter audit |
| `scripts/run_loihi_hw_benchmark.py` | INRC hardware benchmark |
