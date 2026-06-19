# NeuroCUDA

**A PyTorch-to-neuromorphic compiler: ANN→SNN conversion, NIR export, NeuroBench reporting.**

One pipeline. Standard formats. Every number measured and labeled.

---

## Verified Results

### Conversion Accuracy (Gate 3)

| Model | ANN | SNN (T=32) | Gap | Seeds | Test Set |
|-------|-----|------------|-----|-------|----------|
| ResNet-18 / CIFAR-10 | ~95% | ~94% | **0.95% ± 0.14%** | 3 | Full 10K |
| MLP / MNIST | 97.8% | 97.4% | 0.4% | 1 | Full 10K |

### Efficiency (Gate 5 — NeuroBench Algorithm Track)

| Metric | Value | Note |
|--------|-------|------|
| Activation Sparsity | **93.7%** | Measured at T=32, 10K test set |
| Effective Operations | **870M** | vs 15.4B dense — 94% reduction |
| Operations Type | MACs | Graded output (QCFS `[0,λ]` range, not binary) |
| Footprint (float32) | 44.7 MB | GPU / CPU |
| Footprint (8-bit) | 11.2 MB | Modeled for Loihi 2, not silicon |

*Efficiency is from sparsity: 93.7% of activations are zero at each timestep, so 94% of operations are skipped. The IF neurons emit graded values (spike × threshold), so surviving operations count as MACs — the advantage is quantity, not op-type. Eff_ACs = 0.*

### NIR Export (Round-Trip Verified)

| Model | Write | Read | Execute | Accuracy Δ | Status |
|-------|-------|------|---------|------------|--------|
| MLP / MNIST | ✅ | ✅ | ✅ | 0.000000 | Bit-exact |
| ResNet / CIFAR-10 | ✅ | ✅ | ✅ | 0.01% | Functionally exact |

*Full 10K test set, CPU deterministic + GPU, custom Kahn-topology executor (NIRTorch's built-in executor skips branched/residual models).*

### Multi-Backend Validation

| Backend | Spike Deviation | Status |
|---------|----------------|--------|
| GPU (PyTorch) | Reference | Production |
| CPU (PyTorch) | 0/256K | Verified |
| Loihi 2 (simulator) | 0/256K | Bit-accurate |

*Loihi 2: Intel's bit-accurate Lava simulator, NOT physical silicon. Labeled accordingly.*

---

## Install

```bash
git clone https://github.com/neurocuda/neurocuda
cd neurocuda
pip install -r requirements.txt
```

Requirements: `torch>=2.0`, `numpy`, `nir`, `nirtorch`, `neurobench`, `torchvision`

---

## Quickstart

```python
from models import resnet18_cifar, QCFS, build_snn_from_qcfs
from nir_export import export_resnet_to_nir

# Load a QCFS-trained model and convert to SNN
qmodel = resnet18_cifar(lambda: QCFS(L=8))
qmodel.load_state_dict(torch.load("checkpoints/qcfs_resnet18_seed0.pt")["state_dict"])
snn = build_snn_from_qcfs(qmodel)

# Export to NIR (the field's standard SNN format)
nir_graph = export_resnet_to_nir(snn, "my_model.nir")
```

Or run the full NeuroBench report:
```bash
python gate5_neurobench.py --seeds 0 --T 32
```

---

## Reproduce Our Results

Every number in this README is regenerable. The benchmark scripts live in `benchmarks/`.

```bash
# Gate 3: QCFS conversion training (requires ANN checkpoints)
python gate3_qcfs_convert.py --seed 0 --epochs 30

# Gate 5: NeuroBench algorithm-track report
python gate5_neurobench.py --seeds 0 1 2 --T 32

# NIR round-trip verification
python verify_nir_trained.py --seed 0

# Loihi 2 bit-accurate validation
python tests/test_loihi_bitaccurate.py
```

Committed result tables are in `results/`. See `benchmarks/reproduce.sh` for the full reproduction pipeline.

---

## Repository

```
neurocuda/
├── neurocuda/              # Package (in progress)
├── models.py               # ResNet-18, IFNeuron, QCFS
├── nir_export.py           # ANN/NIR export pipeline
├── nir_executor.py         # Custom Kahn-topology NIR executor
├── gate3_qcfs_convert.py   # QCFS conversion training
├── gate5_neurobench.py     # NeuroBench reporting
├── verify_nir_trained.py   # End-to-end NIR verification
├── benchmarks/             # Reproduction scripts
├── results/                # Committed output tables
├── checkpoints/            # Model checkpoints
├── tests/                  # Validation suite
└── examples/               # Demo scripts
```

---

## What This Is

NeuroCUDA is a **systems/tooling contribution**: a single open-source pipeline that ties together PyTorch→SNN conversion, NIR export, and NeuroBench reporting. The individual pieces (QCFS, NIR, NeuroBench) are published work by other groups. What's uncommon is the **integration** — one tool that does all three, verified honestly, with documented limitations.

It is **not** a claim of novel science per component. It is a claim that the neuromorphic ecosystem needs a compiler, and this is one working, honest seed.

---

## Honesty Rules

- Every number measured on full test set (10K images), ≥3 seeds where stated
- Sparsity ≠ accuracy. 93.7% is sparsity, stated separately from accuracy
- Loihi 2 numbers are simulator-validated, labeled "modeled"
- NIR export verified end-to-end (write → read → rebuild → compare)
- "Proven" means the number is reproducible by anyone running the script

---

## Project Status (June 2026)

| Gate | Description | Status |
|------|-------------|--------|
| GATE 1 | Ground truth baselines | ✅ |
| GATE 2 | ANN ResNet-18 ≥93% | ✅ |
| GATE 3 | QCFS converter (0.95% gap) | ✅ |
| GATE 4 | Methods re-tested | ✅ |
| GATE 5 | NeuroBench reporting | ✅ |
| NIR | Round-trip proven | ✅ |
| GATE 6 | Ship | ⬜ |

---

## License

MIT — see [LICENSE](LICENSE)

## Citation

```bibtex
@software{neurocuda2026,
  title = {NeuroCUDA: A PyTorch-to-Neuromorphic Compiler with NIR Export and NeuroBench Reporting},
  year = {2026},
  url = {https://github.com/neurocuda/neurocuda}
}
```
