# INRC Industry Member Proposal

## NeuroCUDA: Multi-Backend SNN Compiler with Loihi 2 Hardware Validation

**Submitted to:** Intel Neuromorphic Research Community (INRC)
**Submitted by:** Quantaracore Technologies LLP
**Date:** June 28, 2026
**Contact:** Krishna Varma, Founder — founder@quantaracore.in
**Website:** https://quantaracore.in/neurocuda
**Member Type:** Industry Member
**Research Vectors:** RV2 (Algorithms) + RV3 (Applications)

## 1. Participants

Krishna Varma — Principal Investigator / Founder — Quantaracore Technologies LLP — founder@quantaracore.in

## 2. Project Abstract

NeuroCUDA is an open-source compiler that converts trained PyTorch models to spiking neural networks and deploys them across GPU, CPU, BrainScaleS-2 analog silicon (Heidelberg), and SpiNNaker-1 digital silicon (Manchester) — through one Python API call. We maintain a Loihi 2 simulator backend validated against published Loihi neuron equations (0 spike deviations per 100K+ test vectors) but have never validated against physical Loihi 2 hardware.

This project closes that gap: deploy converted SNNs to Loihi 2 cloud hardware, validate bit-accurate spike output against our simulator, measure real energy consumption using Intel's Lava energy probes, and produce honest multi-backend NeuroBench benchmarks comparing three distinct physical neuromorphic chips (Loihi 2, SpiNNaker-1, BrainScaleS-2) from one unified compiler pipeline.

## 3. Project Description

### 3.1 The Problem

Spiking neural network research suffers from platform fragmentation. Each neuromorphic hardware system — Loihi 2 (Intel), SpiNNaker-1 (Manchester), BrainScaleS-2 (Heidelberg) — requires its own SDK, its own model format, and its own benchmarking methodology. There is no open-source, unified compiler that takes one trained PyTorch model and deploys it to multiple physical neuromorphic chips with honest, reproducible benchmarks.

### 3.2 What We Already Built

| Backend | Type | Status |
|---|---|---|
| GPU (PyTorch) | Simulator | Production. CUDA-accelerated. |
| CPU (PyTorch) | Simulator | Bit-exact: 0/256K spike devs vs GPU |
| Loihi 2 IF model | Simulator | 0/100K+ spike devs vs published Loihi equations |
| SpiNNaker-1 | Physical silicon | MLP MNIST compiles to sPyNNaker. 5000 core-hr EBRAINS quota approved |
| BrainScaleS-2 | Physical silicon | 138-neuron SNN on chip 57 (Heidelberg). Spikes confirmed 2026-06-28 |
| FPGA | HLS C++ | NIR export pipeline working. Not yet synthesized |

### 3.3 What's Missing — Physical Loihi 2

Our simulator is correct (0 spike deviations) but: never compared to real silicon output, no real energy measurement, no fab variation modeling, cannot honestly claim "physical silicon" per NeuroBench standard.

### 3.4 Approach

Trained PyTorch ANN → QCFS Calibration (5 epochs, learn per-channel thresholds) → IF Neuron Replacement + BPTT Fine-Tuning (5 epochs, surrogate gradients) → NIR Export (vendor-neutral graph) → Lava NIR Importer (Intel's framework) → Loihi 2 Cloud Hardware → Validation: Spike comparison + Energy measurement + NeuroBench report.

Validation sequence: Single IF neuron (bit-identical test) → MLP MNIST (784→256→256→10, 269K params, 97.4% SNN) → CNN N-MNIST (event-camera, 99.88% SNN, 92% sparse) → Multi-seed NeuroBench report (3 seeds, full sets, real energy).

### 3.5 Comparison to Prior Work

| Tool | Multi-Chip | Physical Silicon | NeuroBench | Open-Source |
|---|---|---|---|---|
| SNNToolBox | Partial | Partial | No | Yes |
| Lava-DL | Loihi only | Yes | No | Yes |
| sPyNNaker | SpiNNaker only | Yes | No | Yes |
| NIR | Format only | No | No | Yes |
| NeuroCUDA | 3 chips | 3 chips | Yes | MIT |

### 3.6 Unique Value of Loihi 2

Digital deterministic architecture (bit-reproducible, unlike analog BSS-2). Lava SDK with NIR support (clean Python API, no manual PyNN). Per-synapse weight programming (arbitrary matrices, unlike BSS-2 masks). Built-in energy measurement (Lava probes, real pJ data). 8-bit signed weights (matches NeuroCUDA's quantization pipeline exactly).

### 3.7 Quantitative Evaluation

Spike deviation target: 0 spike time deviations (simulator vs hardware, single neuron). Accuracy gap: ≤2% drop (MLP MNIST: 97.4% → ≥95.4%; CNN NMNIST: 99.88% → ≥97.88%). Real energy: within 2× of 0.08 pJ/SynOp estimate. NeuroBench report: complete 3-chip comparison table.

### 3.8 Definition of Success

A published, reproducible NeuroBench comparison table with verified numbers from three distinct physical neuromorphic chips (Loihi 2, SpiNNaker-1, BrainScaleS-2) measured by one unified open-source compiler pipeline. This would be a unique contribution — no other tool or project currently provides this.

### 3.9 Citations

1. Davies, M. et al. "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning." IEEE Micro, 2018.
2. Orchard, G. et al. "Efficient Neuromorphic Signal Processing with Loihi 2." IEEE ISCAS, 2021.
3. Lava Software Framework. https://github.com/lava-nc/lava
4. NIR — Neuromorphic Intermediate Representation. https://github.com/neuromorphs/NIR
5. Yik, J. et al. "NeuroBench: A Framework for Benchmarking Neuromorphic Computing." Nature Communications, 2025.
6. Bu, T. et al. "Optimal Quantization for SNNs via Calibrated Floor-Shift." NeurIPS, 2023.

## 4. Research Plan

### 4.1 Deliverables

| # | Deliverable | Type | License |
|---|---|---|---|
| D1 | Single-neuron Loihi 2 validation report | Technical report | Public |
| D2 | Lava-based Loihi 2 backend (`loihi2_lava.py`) | Python module | MIT |
| D3 | MLP MNIST + CNN NMNIST Loihi 2 deployment code | Python module | MIT |
| D4 | Multi-backend NeuroBench report (Loihi 2 + SpiNNaker-1 + BrainScaleS-2) | Public benchmark | CC-BY |
| D5 | Tutorial: "Deploy PyTorch to Loihi 2 in One API Call" | Blog + INRC seminar | Public |

### 4.2 Personnel

Krishna Varma — Principal Investigator — Compiler pipeline, Lava integration, benchmark execution, NeuroBench reporting, documentation. Quantaracore Technologies LLP.

### 4.3 Milestones

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | INRC onboarding. Lava configured. Cloud access confirmed. | Setup verified |
| 2 | Single IF neuron on Loihi 2. Spike output validated vs simulator. | D1 complete |
| 3-4 | MLP MNIST deployed via NIR→Lava. Accuracy measured. | D2, D3 partial |
| 5-6 | CNN NMNIST deployed. Multi-seed benchmarks (3 seeds, 10K images). Energy measured. | D3 complete, D4 partial |
| 7-8 | Lava backend complete. `nc.compile(model, target="loihi2_lava")` working. | D2 complete |
| 9-10 | NeuroBench report finalized. Cross-chip table published. INRC seminar. | D4 complete |
| 11-12 | Tutorial published. Code merged to main. Public release. | D5 complete |

### 4.4 Technical Tradeoffs

Quantization: use NeuroCUDA's existing 8-bit per-channel quantization (matches Loihi 2 natively). Network partitioning: MLP MNIST fits single-chip; CNN NMNIST may need 2-chip via Lava's multi-chip compiler. Spike encoding: rate coding (Poisson) for MNIST; event-driven for NMNIST.

## 5. Loihi Resource Needs

Project specifically targets Loihi 2 capabilities. No on-site hardware required.

Maximum network: 269K params (MLP), ~1.5M synapses (CNN). 1-2 chips. No multi-board systems.

Cloud access pattern: occasional interactive sessions (2-3/week, 1-2 hours each). Batch inference (10-20 runs/session, ~100ms each). Total: ~100 core-hours over 12 weeks. Single-system, occasional access sufficient.

Justification: simulator correct but cannot model fab variation, on-chip noise, or real energy. Physical access required for honest "physical silicon" labeling per NeuroBench standard.

## 6. Material Deliverables to INRC

### 6.1 Software Contributions (MIT License)

`neurocuda/backends/loihi2_lava.py` — Lava-based Loihi 2 physical silicon backend. NIR→Lava bridge improvements (if gaps found in existing importer). Energy measurement harness (standardize Lava probe output → NeuroBench JSON). Benchmark scripts and raw data (all publicly reproducible).

### 6.2 Documentation and Knowledge Sharing

"Deploy PyTorch to Loihi 2 in One API Call" — blog post and Jupyter notebook tutorial. Multi-chip NeuroBench report — markdown table and JSON export. INRC Forum seminar — 30-minute presentation and Q&A. Loihi 2 validation technical report — PDF with spike traces and energy logs.

### 6.3 Datasets and Benchmarks

Loihi 2 spike output traces (simulator vs hardware, per-neuron, per-timestep). Energy measurement logs (Lava probe output, per-inference). NeuroBench-format JSON reports (accuracy, latency, energy, sparsity). Comparison tables across all 3 physical chips.

## 7. Intellectual Property

### 7.1 Background IP

NeuroCUDA is MIT-licensed open-source software. All existing code (GPU, CPU, Loihi 2 simulator, SpiNNaker-1, BrainScaleS-2 backends; QCFS converter; NIR exporter; NeuroBench reporter) is already publicly available under the MIT license at https://quantaracore.in/neurocuda. No background IP restrictions.

### 7.2 IP Created Under This Project

All software, documentation, and benchmark data created under this project will be released as MIT-licensed open-source to the public domain. Quantaracore Technologies LLP does not seek to retain proprietary rights over INRC project outputs.

### 7.3 Agreement Type

Corporate Participation Agreement requested. Please send to founder@quantaracore.in for review and execution by Quantaracore Technologies LLP.

### 7.4 Funding

No Intel funding requested. Loihi 2 cloud hardware access only.

## 8. Submission

**Applicant:** Krishna Varma, Founder, Quantaracore Technologies LLP. founder@quantaracore.in. https://quantaracore.in/neurocuda. Member Type: Industry Member. Research Vectors: RV2 (Algorithms) + RV3 (Applications).

Submitted via email to inrc_interest@intel.com (online Qualtrics form reported inactive). Supporting materials: NeuroCUDA GitHub, multi-backend validation (5 backends, 3 simulators + 2 physical silicon), BrainScaleS-2 chip 57 confirmation (June 28, 2026), SpiNNaker-1 EBRAINS quota (5000 core-hours), NeuroBench compliance.

*Submitted June 28, 2026 by Krishna Varma on behalf of Quantaracore Technologies LLP.*
