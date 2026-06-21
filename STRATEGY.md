# NeuroCUDA — Comprehensive Strategic Plan (June 2026)

> **Methodology**: Based on research across 30+ sources covering market forecasts, hardware landscape, software ecosystem, competitive analysis, and developer pain points. All data from 2025-2026 publications.

---

## 1. MARKET LANDSCAPE

### Market Size & Growth

| Year | Conservative | Mid-Range | Aggressive |
|------|-------------|-----------|------------|
| 2025 | $2.6B | $5.5B | $6.4B |
| 2030 | $5.0B | $15.0B | $25.0B |
| 2035 | $15.0B | $35.0B | $61.5B |
| **CAGR** | **16-22%** | **26-33%** | **50-90%** |

Note: Wide variance depends on scope — chip-only ($50-340M) vs full ecosystem ($2.8-6.4B). Software is the fastest-growing segment.

### Key Growth Drivers
1. **Energy imperative**: SNNs deliver 10-500× energy savings vs GPUs at the edge
2. **Edge AI explosion**: Always-on inference in cameras, wearables, industrial sensors
3. **Hardware maturing**: Loihi 2, Akida, Speck, Pulsar all shipping commercially
4. **Government funding**: $5B+ in VC/government funding flowing into non-GPU architectures (2024-2026)
5. **Talent pipeline**: Neural engineering programs growing globally

### Where The Money Is
- **Hardware**: 65% of current revenue (chips, processors, sensors)
- **Software**: Fastest growing (SNN toolchains, deployment, cloud services)
- **Services**: Smallest but growing (integration, consulting, training)
- **Leading verticals**: Automotive, Defense, Healthcare, Industrial IoT

---

## 2. HARDWARE ECOSYSTEM — Every Major Platform

### Commercial (Shipping in Volume)

| Platform | Company | Type | Scale | Power | Cost | Status |
|----------|---------|------|-------|-------|------|--------|
| **Loihi 2** | Intel | Digital neuromorphic | 1M neurons/chip, 120M synapses | ~30 mW (sparse) | Research access | Research flagship |
| **Hala Point** | Intel | 1,152× Loihi 2 | 1.15B neurons | System-level | Sandia Labs | Largest neuromorphic system |
| **Akida AKD1500** | BrainChip | Digital SNN accelerator | Event-driven | Ultra-low | $200-500 dev kit | First to commercialize (2021) |
| **Speck** | SynSense | Sensing+computing SoC | Async event-driven | 0.42 mW idle, 0.70 mW dynamic | $100-300 | Shipping in IoT |
| **Xylo** | SynSense | Audio neuromorphic | Real-time audio | <1 mW | Dev kit | Shipping |
| **Pulsar** | Innatera | Neuromorphic MCU | Analog+digital SNN engines + RISC-V | <1 mW inference | **Under $5** | Volume production 2025 |
| **SpiNNaker2** | SpiNNcloud | Digital, fully programmable | 152 ARM cores/chip, 5M+ cores in largest system | Event-driven | Research/Enterprise | Deployed at Sandia, Leipzig, UTSA |

### Research / Pre-Commercial

| Platform | Company | Type | Note |
|----------|---------|------|------|
| **TrueNorth** | IBM | Digital neuromorphic | Foundational, not actively commercialized |
| **NorthPole** | IBM | Neuromorphic principles, non-spike | R&D, strong performance claims |
| **GrAI One** | GrAI Matter Labs | Event-driven AI processor | Vision AI focus |
| **Memristor-CIM** | Various | Analog in-memory | TRL 2-5, 10 fabricated prototypes |
| **Great Sky** | Great Sky (2026) | Superconducting optoelectronic | Highest ceiling, highest risk |

### The Critical Hardware Takeaway
- **7+ different chips, 7+ different SDKs, ZERO cross-platform tools**
- Every deployment requires custom engineering
- This fragmentation is the #1 barrier to industry adoption
- **The market is screaming for a unified deployment layer**

---

## 3. SOFTWARE ECOSYSTEM — Every Major Tool

### Training Frameworks

| Tool | What It Does | Strengths | Weaknesses |
|------|-------------|-----------|------------|
| **snnTorch** | SNN training (PyTorch) | Mature, well-documented, surrogate gradients | Training only. No conversion. No deployment. |
| **SpikingJelly** | SNN training library | Fast, Chinese ecosystem | Training only. No conversion. English docs limited. |
| **Sinabs** | SNN library (SynSense) | Event-driven, good for Speck hardware | Vendor-locked to SynSense. Not portable. |

### Deployment Frameworks

| Tool | What It Does | Strengths | Weaknesses |
|------|-------------|-----------|------------|
| **Lava** | SNN framework (Intel) | Powerful, Loihi-optimized | Loihi-specific. Complex. No conversion pipeline. |
| **SNNToolBox** | ANN→SNN conversion | Pioneering, supports Keras/PyTorch | OUTDATED. No modern QCFS methods. No multi-backend. Not maintained. |
| **Rockpool** | SNN deployment | Multi-platform | Small community. Limited conversion support. |
| **MetaTF** | BrainChip Akida SDK | Akida-optimized | Vendor-locked. Proprietary. |

### Standards & Infrastructure

| Tool | What It Does | Strengths | Weaknesses |
|------|-------------|-----------|------------|
| **NIR** | Neuromorphic IR | Standard format, cross-platform | Format only. No training/conversion/execution for complex models. |
| **Synfire** | Community model hub | NEW (March 2026). Built on NIR. Vendor-neutral. HuggingFace for SNNs. | JUST launched. No models yet. No conversion pipeline. |
| **NeuroBench** | Benchmark standard | Standardized metrics | Early adoption. Limited model coverage. |

### Developer Experience Tools — COMPLETELY MISSING

| What ML Engineers Have (ANN) | Neuromorphic Equivalent | Status |
|------------------------------|------------------------|--------|
| TensorBoard | Nothing | **MISSING** |
| Weights & Biases / MLflow | Nothing | **MISSING** |
| HuggingFace Hub (500K+ models) | Synfire (just launched, 0 models) | **NASCENT** |
| ONNX (universal model interchange) | NIR | **EARLY** |
| MLOps (CI/CD, versioning) | Nothing | **MISSING** |
| `pip install transformers` (one line) | Nothing | **MISSING** |
| ROS2 AI packages | Nothing integrated | **MISSING** |

---

## 4. COMPETITIVE POSITIONING — Where NeuroCUDA Fits

### The Stack — Who Owns What

```
┌─────────────────────────────────────────────────────────┐
│  APPLICATION    │  Drones, robots, IoT, auto, defense   │
├─────────────────────────────────────────────────────────┤
│  DEPLOYMENT     │  ROS2 nodes, edge runtimes            │ ← NO ONE OWNS THIS
├─────────────────────────────────────────────────────────┤
│  MODEL HUB      │  Synfire (Innatera) — just launched   │ ← NASCENT
├─────────────────────────────────────────────────────────┤
│  BENCHMARKS     │  NeuroBench, custom per-paper         │ ← FRAGMENTED
├─────────────────────────────────────────────────────────┤
│  CONVERSION     │  NEUROCUDA ← ONLY pip-installable     │ ← WE OWN THIS
│  PIPELINE       │  solution with verified results       │
├─────────────────────────────────────────────────────────┤
│  TRAINING       │  snnTorch, SpikingJelly, PyTorch      │ ← MATURE
├─────────────────────────────────────────────────────────┤
│  HARDWARE       │  Loihi 2, Akida, Speck, Pulsar, ...  │ ← MATURE
└─────────────────────────────────────────────────────────┘
```

### NeuroCUDA's Unique Position

| What | NeuroCUDA | Closest Competitor | Gap |
|------|-----------|-------------------|-----|
| One-line ANN→SNN conversion | ✅ `nc.convert()` | SNNToolBox (outdated, not maintained) | **We own this** |
| Multi-backend deployment | ✅ GPU, CPU, Loihi, FPGA stubs | Lava (Loihi-only) | **We own this** |
| NIR export | ✅ Working, verified | NIR reference tooling (no conversion) | **We own this** |
| Model hub with real weights | ✅ 4 models, HuggingFace | Synfire (0 models, just launched) | **We're ahead** |
| Reproducible benchmarks | ✅ `reproduce.py`, full test sets, ≥3 seeds | NeuroBench (early) | **We own this** |
| pip install | ✅ `pip install neurocuda` | Lava, SpikingJelly (complex install) | **We own this** |
| ROS2 integration | ❌ Not built | Nothing (everyone builds custom) | **Open opportunity** |
| Streaming inference | ❌ Not built | Nothing | **Open opportunity** |
| Physical hardware validation | ❌ Simulator only | Lava, MetaTF (vendor-specific) | **Gap to fill** |

---

## 5. THE 5 PLAYS — Ranked by Impact × Feasibility

### PLAY 1: ROS2 NeuroCUDA Node 🥇
**Impact: VERY HIGH | Feasibility: HIGH | Time: 2-3 weeks**

**What**: One ROS2 package. Any robot, any drone — add SNN perception/control with one command.

```bash
ros2 run neurocuda infer --model cnn-nmnist-snn --topic /dvs/events
ros2 run neurocuda control --model dqn-cartpole-snn --topic /cmd_vel
```

**Why this wins**:
- Event camera + SNN papers are publishing NOW in 2026
- UAV inspection delivers 90%+ energy reduction (proven on Loihi 2)
- Drones can fly hours instead of minutes on battery
- Every project (Eldarin, TU Delft, LENS) builds custom — none ships a reusable package
- Defense contractors need this YESTERDAY (Sandia, DARPA-funded)

**Market Pull**:
- Multi-UAV neuromorphic control published (15 indoor + 6 outdoor UAVs)
- 312× energy savings demonstrated on real drone workloads
- $12.5B drone market by 2026, all needing perception/control

**Users**: Every robotics lab. Every drone company. Every defense contractor.

---

### PLAY 2: Synfire Integration — Feed Their Model Hub 🥈
**Impact: VERY HIGH | Feasibility: HIGH | Time: 1-2 weeks**

**What**: NeuroCUDA becomes the OFFICIAL conversion pipeline for Synfire.

Every model on Synfire needs to come from somewhere. NeuroCUDA's `convert()` is the only pip-installable, verified ANN→SNN pipeline. Integration means:
- `nc.convert()` → export to Synfire format → publish to their hub
- Every Synfire user discovers NeuroCUDA
- Network effect: converters make models → models attract users → users demand more converters

**Why this wins**:
- Synfire launched March 2026 — it's brand new, building its ecosystem
- They NEED models. We MAKE models.
- Built on NIR (which we already export to)
- Steve Furber (ARM co-creator, SpiNNaker) endorsed Synfire
- First-mover advantage — be the FIRST pipeline integrated

**Timing**: RIGHT NOW. Synfire just launched. Window is wide open.

---

### PLAY 3: Real-Time Streaming Inference 🥉
**Impact: HIGH | Feasibility: HIGH | Time: 1-2 weeks**

**What**: `nc.stream(source="/dev/dvs0", model=snn)` — continuous event-driven inference with stateful IF neurons.

**Why this wins**:
- Current tools process events as BATCHED FRAMES — losing temporal precision
- 2026 research: "latency degrades online accuracy by >50%" with frame-based eval
- NeuroCUDA already has per-frame spiking loop with auto-detection — streaming is a natural extension
- True neuromorphic processing keeps IF membrane state alive across events
- Enables sub-millisecond latency (critical for drone control)

**Users**: Event camera users, real-time systems, robotics, autonomous vehicles.

---

### PLAY 4: Public Benchmark Leaderboard
**Impact: HIGH | Feasibility: MEDIUM | Time: 3-4 weeks**

**What**: neurocuda.dev/benchmarks — THE trusted independent benchmark for SNN conversion quality.

**Why this wins**:
- No standardized cross-platform benchmark exists
- Every paper cherry-picks results differently
- 2026 research confirms: "Benchmarking is the #1 gap after tooling"
- NeuroCUDA already has: honesty rules, full test sets, ≥3 seeds, `reproduce.py`
- Being the benchmark = being the standard = being cited by every paper

**What to build**:
1. Leaderboard website (GitHub Pages, sortable table)
2. Standardized evaluation harness (any model → score)
3. Auto-validation (submit → we convert → we benchmark → publish)
4. Monthly reports with methodology

---

### PLAY 5: Hardware-in-the-Loop Validation
**Impact: CRITICAL (credibility) | Feasibility: MEDIUM | Time: 3-6 months**

**What**: Run NeuroCUDA models on REAL Loihi 2 and FPGA hardware with measured power/latency.

**Why this wins**:
- Currently "simulator-validated only" — limits credibility
- Every competitor (Lava, MetaTF) has hardware validation
- Physical measurements transform NeuroCUDA from "research tool" to "deployment tool"
- Required for defense/industrial contracts

**Hardware Needed**: Loihi 2 dev kit (Intel NRC access), FPGA board

---

## 6. PRIORITY ROADMAP

```
PHASE 1 (June-July 2026)          PHASE 2 (Aug-Sept 2026)        PHASE 3 (Oct-Dec 2026)
┌─────────────────────┐          ┌─────────────────────┐        ┌─────────────────────┐
│ ✅ PyPI live        │          │ ROS2 Node           │        │ Hardware Validation  │
│ ✅ HuggingFace hub  │    →     │ Streaming Inference │   →   │ Synfire Integration  │
│ ✅ 4 models w/ wgts │          │ Benchmark Site      │        │ Multi-Backend HW     │
│ 🔲 ROS2 Node        │          │ 10+ models on HF    │        │ Enterprise Features  │
└─────────────────────┘          └─────────────────────┘        └─────────────────────┘
  WHAT WE HAVE NOW                  NEXT 30 DAYS                    REST OF 2026
```

---

## 7. THE ENDGAME — What NeuroCUDA Becomes

```
BEFORE:                            AFTER:
"Train in PyTorch"                 "pip install neurocuda"
"Convert with academic scripts"    "nc.convert(model, data)"
"Deploy? Figure it out yourself"   "nc.compile(target='loihi')"
"No standard benchmarks"           "neurocuda.dev/benchmarks"
"No pre-trained SNNs"              "nc.hub.load('cnn-nmnist-snn')"
"No ROS integration"               "ros2 run neurocuda infer"
```

### The 3-Word Strategy

**Convert. Deploy. Reproduce.**

One tool that does what currently takes 4 different tools and custom engineering.

### Why NeuroCUDA Wins

| Advantage | Why It Matters |
|-----------|---------------|
| **First to PyPI** | Zero friction. `pip install` → done. |
| **Honest numbers** | Trust in a field of cherry-picking. |
| **Multi-backend** | Not tied to any vendor. Future-proof. |
| **NIR native** | Industry standard. Interoperable. |
| **Open source** | Community builds on it. |
| **ROS2 ready** (next) | Opens $12.5B drone + $20B robotics markets. |

### The Moat

Once NeuroCUDA is the standard conversion+deployment pipeline:
- Every neuromorphic paper cites it
- Every hardware vendor wants compatibility
- Every robotics project depends on it
- Network effects make it hard to displace

This is the NVIDIA CUDA playbook, applied to neuromorphic computing.

---

## 8. IMMEDIATE NEXT ACTIONS (This Week)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 1 | **Build ROS2 NeuroCUDA node** — `ros2 run neurocuda infer` | 1 week | Makes NeuroCUDA instantly useful to every robotics developer |
| 2 | **Streaming inference** — `nc.stream(source, model)` | 3 days | Unlocks real event-camera processing |
| 3 | **Contact Synfire team** — propose integration | 1 day | First-mover advantage on their platform |
| 4 | **Export remaining model weights** — CartPole, MNIST, CIFAR | 2 days GPU | Complete the model hub |
| 5 | **Leaderboard website** — `neurocuda.dev` | 3 days | Become the trusted benchmark |

---

## Sources

- ResearchAndMarkets.com — Neuromorphic Computing Market Report 2025-2035
- Meticulous Research — Global Neuromorphic Computing Forecast 2026-2036
- Innatera — Synfire Platform Launch (March 2026)
- Innatera — Pulsar Neuromorphic MCU Specifications (May 2025)
- SpiNNcloud — SpiNNaker2 Deployments (Sandia, Leipzig, UTSA, 2025)
- DTU/ETH Zurich — SNN Object Detection on Loihi 2 (Neurocomputing, 2026)
- ACM — Developer Tool Gaps in Neuromorphic HCI (April 2025)
- Oulu University — Neuromorphic Deployment Toolchain Gaps (2025)
- Enotrium/Eldarin — ROS2 + SNN Drone Navigation (GitHub, 2025)
- TU Delft — Event-Based Planar Optical Flow for Drones (GitHub, 2025)
- arXiv:2502.05938 — DVS Drone Navigation Framework (IJCNN 2025)
- arXiv:2602.02439 — NeuEdge: Adaptive SNN Framework for Edge AI (2026)
- ESTU — Spiking Transformer on Lattice FPGA (GitHub, 2025)
