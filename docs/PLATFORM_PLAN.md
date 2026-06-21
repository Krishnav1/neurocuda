# NeuroCUDA Platform — Website, Demo, Dashboard

> Based on research of 100+ dev tool landing pages and HuggingFace Spaces best practices (2025-2026)

---

## The 3 Pieces

```
neurocuda.dev          HuggingFace Space         GitHub/HF Hub
(Landing Page)         (Live Demo)               (Models + Code)
     │                      │                         │
     │  "Try it →"  ──────→ │  Upload model,          │
     │                      │  see SNN results        │
     │                      │       │                  │
     │  ← "View models" ────┴───────┘                  │
```

---

## 1. neurocuda.dev — Landing Page (GitHub Pages, Free)

### Hero Section
```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Convert PyTorch Models to Spiking Neural          │
│   Networks. One Line.                               │
│                                                     │
│   ┌─────────────────────────────────────────────┐   │
│   │ $ pip install neurocuda                      │   │
│   │ $ python                                      │   │
│   │ >>> import neurocuda as nc                    │   │
│   │ >>> snn, stats = nc.convert(model, data)     │   │
│   │ >>> print(stats['if_accuracy'])               │   │
│   │ 99.88%                                        │   │
│   └─────────────────────────────────────────────┘   │
│                                                     │
│   [Try Live Demo →]  [GitHub ⭐]  [pip install]     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Why this works**: Code snippet > marketing text. Developers trust what they can read and copy. The `99.88%` number is the hook.

### Trust Block (Right Below Hero)
```
┌──────────────────────────────────────────────────┐
│  PyPI Downloads    GitHub Stars    HF Models      │
│    1,234+             50+            8           │
│                                                  │
│  "First pip-installable ANN→SNN compiler"        │
│  Used by researchers at [logos as we get them]   │
└──────────────────────────────────────────────────┘
```

### Before vs After (Problem-Oriented)
```
┌────────────────────────────┐    ┌──────────────────────────┐
│         BEFORE             │    │          AFTER            │
│                            │    │                          │
│  Custom scripts per model  │ →  │  nc.convert() — one line │
│  Vendor SDK for each chip  │ →  │  Multi-backend: GPU/CPU/ │
│                            │    │  Loihi/FPGA              │
│  Cherry-picked results     │ →  │  Full test sets, 3 seeds │
│  No pre-trained models     │ →  │  Pre-converted on HF Hub │
│  Months of engineering     │ →  │  60 seconds              │
└────────────────────────────┘    └──────────────────────────┘
```

### Comparison Table (Names Competitors)
```
┌──────────────┬───────────┬──────────┬────────────┬──────────┐
│ Feature      │ NeuroCUDA │ SNNToolBox│ Lava       │ SpkJelly │
├──────────────┼───────────┼──────────┼────────────┼──────────┤
│ pip install  │    ✅     │    ❌    │    ❌      │   ❌     │
│ One-line API │    ✅     │    ❌    │    ❌      │   ❌     │
│ CS-QCFS      │    ✅     │    ❌    │    ❌      │   ❌     │
│ BPTT FT      │    ✅     │    ❌    │    ❌      │   ✅     │
│ Multi-backend│    ✅     │    ❌    │ Loihi only │   ❌     │
│ NIR export   │    ✅     │    ❌    │    ❌      │   ❌     │
│ HF Model Hub │    ✅     │    ❌    │    ❌      │   ❌     │
│ ROS2 package │    ✅     │    ❌    │    ❌      │   ❌     │
│ Benchmarks   │ Honest    │ Outdated │   None     │   None   │
└──────────────┴───────────┴──────────┴────────────┴──────────┘
```

### Benchmark Leaderboard (Interactive)
```
┌────────────────────────────────────────────────────────────┐
│  SNN Conversion Leaderboard                                 │
│                                                             │
│  Model             ANN     SNN     Gap    Sparsity  Method  │
│  ────────────────  ─────   ─────   ────   ────────  ──────  │
│  CNN NMNIST        99.70   99.88  -0.18   91.7%    CS-QCFS │
│  ResNet-18 CIFAR   95.56   94.61   0.95   93.7%    QCFS→IF │
│  MLP MNIST         97.80   97.40   0.40    —        QCFS→IF │
│  StrongCNN CIFAR   80.40   74.30   6.10    —        QCFS+FT │
│                                                             │
│  [Submit your model to be benchmarked →]                    │
└────────────────────────────────────────────────────────────┘
```

### Sections in order:
1. **Hero** — Code + CTA
2. **Trust** — Numbers + social proof
3. **Before/After** — Pain → Solution
4. **Demo GIF** — Animated terminal recording
5. **Comparison** — vs competitors (honest)
6. **Leaderboard** — Live benchmarks
7. **Quickstart** — 3-code-block tutorial
8. **Community** — Discord, GitHub, blog
9. **Footer** — Links, license, citation

---

## 2. HuggingFace Space — Live Demo (Free GPU, Gradio)

### What It Does
Someone uploads a PyTorch model (or selects from examples) → the Space converts it → shows:
- SNN accuracy
- Spike raster visualization  
- Energy estimate
- Download link for NIR file

### The Interface
```
┌─────────────────────────────────────────────┐
│  NeuroCUDA Live Demo 🧠                      │
│                                              │
│  Step 1: Select Model                        │
│  ┌─────────────────────────────────────┐     │
│  │ ○ Upload .pt file                   │     │
│  │ ○ Try example: MNIST MLP            │     │
│  └─────────────────────────────────────┘     │
│                                              │
│  Step 2: Convert                             │
│  [Convert to SNN →]                          │
│                                              │
│  Step 3: Results                             │
│  ┌─────────────────────────────────────┐     │
│  │ ANN Accuracy:  97.80%               │     │
│  │ SNN Accuracy:  97.40%               │     │
│  │ Gap:           0.40%                │     │
│  │ Sparsity:      68.5%                │     │
│  │                                              │
│  │ Spike Activity: ████░░░░░░ 42%      │     │
│  │ Layer 1:        ███░░░░░░░ 31%      │     │
│  │ Layer 2:        ██████░░░░ 58%      │     │
│  └─────────────────────────────────────┘     │
│                                              │
│  [Download NIR →]  [Download Model →]        │
└─────────────────────────────────────────────┘
```

### Technical
```python
# app.py — runs on HuggingFace Space with free GPU
import gradio as gr
import neurocuda as nc

def convert_and_show(model_file, example_name):
    model = load_model(model_file or example_name)
    snn, stats = nc.convert(model, train_loader)
    
    return {
        "ANN Accuracy": stats.get("ann_accuracy", "N/A"),
        "SNN Accuracy": stats["if_accuracy"],
        "Gap": f"{stats['gap']:.2f}%",
        "Sparsity": f"{stats.get('sparsity', 'N/A')}%",
    }

gr.Interface(fn=convert_and_show, ...).launch()
```

---

## 3. What To Build First

| Priority | Piece | Time | Cost | Impact |
|----------|-------|------|------|--------|
| **1** | neurocuda.dev landing page | 1 day | Free (GitHub Pages) | HIGH |
| **2** | HuggingFace Space demo | 1 day | Free (HF GPU) | VERY HIGH |
| **3** | Leaderboard page | 1 day | Free (static data) | HIGH |
| **4** | Blog + tutorials | Ongoing | Free (Medium/Dev.to) | MEDIUM |
| **5** | Newsletter | Ongoing | Free (Substack) | MEDIUM |

---

## Tech Stack (Everything Free)

| Piece | Technology | Cost |
|-------|-----------|------|
| Landing page | GitHub Pages + Jekyll/Next.js static | $0 |
| Domain | neurocuda.dev (if you want) | ~$12/year |
| Live demo | HuggingFace Spaces (Gradio) | $0 (free GPU) |
| Model hub | HuggingFace Model Hub | $0 |
| Leaderboard | Static JSON + GitHub Pages | $0 |
| Blog | Medium / Dev.to | $0 |
| Analytics | Plausible (self-hosted) or GoatCounter | $0 |
| Community | Discord (you have) + GitHub Discussions | $0 |

---

## The Complete Architecture

```
User visits neurocuda.dev
    │
    ├─→ Reads hero, sees code snippet
    ├─→ Clicks "Live Demo" → HuggingFace Space
    │       │
    │       ├─→ Uploads their model
    │       ├─→ Space converts it via NeuroCUDA
    │       └─→ Shows results, download NIR
    │
    ├─→ Browses Leaderboard → sees comparisons
    ├─→ Clicks "Models" → HuggingFace Hub
    │       └─→ Downloads pre-converted SNN
    │
    ├─→ Clicks "Docs" → GitHub README / Wiki
    └─→ Clicks "Community" → Discord
```

No backend. No servers. No database. Everything free. Everything real.
