# CLAUDE.md — NeuroCUDA

## PRIME DIRECTIVE — HONESTY RULES (read before every task)

1. **A failed run is a bug, never a "finding."** If a published method produces bad results, your implementation is broken. Investigate. Do not claim you discovered the method doesn't work.
2. **Full test set only.** CIFAR-10 = 10,000 images. Never report 500-image subsets as results.
3. **≥3 seeds.** Every number is mean ± std. Single runs are not results.
4. **Label hardware precisely.** "Loihi 2 simulator validated against Lava" — never "Loihi 3" or "silicon" unless physically run on it.
5. **Gate failure = STOP.** Do not proceed. Do not relabel the target.
6. **Report failures first.** "Gate 2 FAILED. Cause: X. Options: Y."
7. **No marketing language.** No "world-class," "nobody has done this," "🔥." Just measurements.

## Project Identity

**NeuroCUDA is a systems/tooling contribution.** A pip-installable compiler that takes PyTorch models and deploys them across GPU, CPU, Loihi 2 simulator, and FPGA through one API call.

**Goal:** Build an honest, working seed. NOT beat SOTA accuracy. NOT claim novel science.

## Current State (June 19, 2026)

### Real Numbers (full test set, need multi-seed verification)
- MLP/MNIST: ANN 97.8%, SNN 97.4%, gap 0.4% (1 seed)
- StrongCNN/CIFAR-10: ANN 80.4%, SNN 74.3%, gap 6.0% (1 seed, convert + FT)
- SEW-ResNet/CIFAR-10: 67.7% at T=8 (direct SNN, 50 epochs, 1 seed)
- ResNet/CIFAR-10: ANN 92.1%, SNN 70.1%, gap 22.0% (convert + FT, 1 seed)

### What Works
- Multi-backend deployment: GPU/CPU/Loihi ≤1.2% deviation
- Loihi 2 bit-accurate validation: 0/256K spike deviations
- BN folding: lossless
- Post-conversion fine-tuning: +7-52% gain

### Known Bugs
- QCFS: λ frozen at ~1.0 for layers 1-2 (gradient/learning-rate bug)
- Weight normalization: destroys ANN accuracy (implementation error)
- "Spike-density trap" and "QCFS non-generalization" are bugs, not discoveries

## The Gates

### GATE 1 — Ground Truth
Re-measure everything on full test set, 3 seeds. Produce honest baseline table.

### GATE 2 — Fix Base ANN
Train ResNet-18 on CIFAR-10 to ≥93% (solved problem, standard recipe).

### GATE 3 — Fix Converter (QCFS)
Initialize from pretrained ANN. Separate higher LR on λ. Gap ≤5%.

### GATE 4 — Re-test Methods
Re-run per-channel, SPIKE-NORM, weight-norm on fixed pipeline.

### GATE 5 — NeuroBench
Standard-format, multi-seed, multi-backend reporting.

### GATE 6 — Ship
Clean README, reproducible benchmarks, honest paper.

## NON-Goals
- Do NOT chase SOTA accuracy
- Do NOT claim physical silicon without physical silicon
- Do NOT describe bugs as discoveries
- Do NOT add scope until Gates 1-6 pass

## Reporting Format
After each gate:
```
GATE N — PASS/FAIL
Headline: <metric> = <mean> ± <std> (3 seeds, full test set)
Target: <target>
Changes: <1-3 bullets>
Surprises: <honest>
Verified: YES/NO
Next: YES/NO — if NO, why
```