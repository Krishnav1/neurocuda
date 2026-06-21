# NeuroCUDA × Synfire — Integration Plan

## Research Summary (June 22, 2026)

### What Synfire Is

Synfire is an **open registry for neuromorphic models** launched by Innatera in March 2026. Think "HuggingFace for spiking neural networks." It's built on the NIR standard and is vendor-neutral.

**Architecture:**
```
┌──────────────────────────────────────────────┐
│  synfire.dev (Web UI)                        │
│  - Search, discover, browse models           │
│  - View model cards, metadata, hardware tags │
├──────────────────────────────────────────────┤
│  synfire CLI (pip install synfire)           │
│  - synfire push org/model --version 1.0.0   │
│  - synfire pull org/model                    │
│  - synfire search "classifier"               │
├──────────────────────────────────────────────┤
│  Python SDK (from synfire import Client)     │
│  - client.push(...), client.pull(...)        │
│  - client.search(...)                        │
└──────────────────────────────────────────────┘
```

### Key Facts

| Fact | Detail |
|------|--------|
| **Launched** | March 25, 2026 (Edge AI San Diego) |
| **pip package** | `synfire` — NOT YET ON PYPI (early access) |
| **Current state** | Platform live, search page online, **ZERO models published** |
| **Format** | NIR (`.nir` files) + `nir-card.json` metadata |
| **Hardware targets** | Pulsar, Loihi2, SpiNNaker2, BrainScaleS-2, Xylo, Speck, Generic |
| **Versioning** | Semantic (1.0.0), immutable releases, SHA256 checksums |
| **Auth** | Browser login + API tokens (`SYNFIRE_TOKEN`) |
| **Endorsed by** | Steve Furber (ARM co-creator, SpiNNaker) |

### What Publishing Requires

```
synfire push <org>/<repo> --version 1.0.0 \
  --model ./model.nir \        # NIR format model
  --card ./nir-card.json       # Metadata (license required)
```

**nir-card.json format:**
```json
{
  "license": "MIT",                    // REQUIRED — SPDX identifier
  "capabilities": ["Classification"],  // Optional
  "tested_platforms": ["Pulsar"],     // Optional
  "tags": ["vision", "snn"],          // Optional
  "description": "...",               // Optional
  "authors": [{"name": "..."}],       // Optional
  "links": [{"url": "..."}]           // Optional
}
```

**Auto-extracted from model.nir**: Neuron types (IF, LIF), operations, node count, parameter count, size category (tiny/small/medium/large).

---

## The Opportunity

### Why NOW is the Time

1. **Synfire has ZERO models.** The platform launched 3 months ago and the search page is empty.
2. **The pip package isn't on PyPI yet** — still in early access.
3. **No one has published a conversion pipeline integration.**
4. **NeuroCUDA already exports to NIR** (verified, 0.000000 max abs diff).
5. **NeuroCUDA already has 4 models with real weights on HuggingFace.**

### First-Mover Advantages

- Be the **FIRST** organization to publish models on Synfire
- Be the **FIRST** conversion pipeline integrated with Synfire
- Establish `neurocuda` as the default namespace before anyone else
- Every future model publisher sees NeuroCUDA as the pioneer
- Network effect: models attract users → users publish more models → NeuroCUDA is the pipeline

### The Play

**NeuroCUDA becomes the #1 publisher on Synfire AND the standard way to generate NIR models for Synfire.**

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  PyTorch    │ →   │  NeuroCUDA   │ →   │  Synfire    │
│  ANN Model  │     │  nc.convert()│     │  Registry   │
└─────────────┘     │  nc.to_nir() │     └─────────────┘
                    └──────────────┘
```

---

## Implementation Plan

### Phase 1: Publish 4 Models to Synfire (Today)

We already have everything needed:
- ✅ 4 models with verified weights
- ✅ NIR export working (`nc.to_nir()`)
- ✅ Model cards (on HuggingFace)
- ✅ Verified accuracy, sparsity, energy numbers

**What to build**: A script that:
1. Loads each converted SNN model
2. Exports to NIR format
3. Creates `nir-card.json` from model metadata
4. Publishes to Synfire using their SDK/CLI

```python
# neurocuda/synfire_publisher.py
import neurocuda as nc
from synfire import Client

def publish_to_synfire(model_name, version="1.0.0"):
    # 1. Load model
    snn, info = nc.hub.load(model_name)

    # 2. Export to NIR
    nir_graph = nc.to_nir(snn, T=16, model_name=model_name)

    # 3. Create nir-card.json
    card = create_nir_card(info)

    # 4. Push to Synfire
    client = Client()
    client.push(
        repository=f"neurocuda/{model_name.split('/')[-1]}",
        version=version,
        model_path=save_nir(nir_graph),
        card_path=save_card(card),
        release_notes=generate_release_notes(info),
    )
```

### Phase 2: Auto-Publish Pipeline (1 Week)

Add to CI/CD or export script:
- Every `nc.convert()` can auto-publish to Synfire
- `python scripts/export_hub_models.py --publish-to-synfire`
- Models appear on both HuggingFace AND Synfire

### Phase 3: Synfire-First NIR Export (1 Week)

Optimize NIR export for Synfire compatibility:
- Add `tested_platforms` metadata
- Auto-detect hardware compatibility
- Generate Synfire-optimized model cards
- Validate NIR against Synfire schema

---

## Models to Publish

| # | Synfire Repo | HuggingFace | Accuracy | Sparsity | Status |
|---|-------------|-------------|----------|----------|--------|
| 1 | `neurocuda/cnn-nmnist-snn` | ✅ Live | 99.95% IF | 91.9% | ✅ Ready |
| 2 | `neurocuda/robotics-perception-snn` | ✅ Live | 99.85% IF | 92.6% | ✅ Ready |
| 3 | `neurocuda/mlp-mnist-snn` | ✅ Live | 96.19% IF | — | ✅ Ready |
| 4 | `neurocuda/dqn-cartpole-snn` | ✅ Live | ANN→LIF | 68.5% | ✅ Ready |
| 5 | `neurocuda/resnet18-cifar10-snn` | Card only | 94.61% | 93.7% | 🔲 Needs weights |

---

## Competitive Analysis

### Who Could Also Publish on Synfire

| Potential Publisher | What They Have | Why NeuroCUDA Wins |
|--------------------|---------------|-------------------|
| **Intel (Lava)** | Loihi-optimized SNNs | Vendor-specific, not pip-installable, no conversion |
| **SynSense (Sinabs)** | Speck-optimized SNNs | Vendor-specific, small community |
| **Academic labs** | Individual research models | No standardized pipeline, one-off scripts |
| **snnTorch community** | Direct-trained SNNs | Training only, no conversion, no NIR export built-in |
| **NeuroCUDA** | **Automated conversion + NIR + verified benchmarks** | **Only end-to-end pipeline** |

### The Moat

Synfire needs content. NeuroCUDA is the only tool that can generate NIR models **at scale** via automated conversion. Every model published through NeuroCUDA's pipeline increases the moat.

---

## What We Need to Build

### 1. Synfire Publisher Module (Priority: TODAY)

```python
# neurocuda/synfire.py
def publish(model, repo, version, hardware_targets, license="MIT"):
    """Publish a NeuroCUDA SNN to the Synfire registry."""
    nir_graph = to_nir(model)
    card = make_nir_card(model.metadata, hardware_targets, license)
    client = Client(token=os.environ["SYNFIRE_TOKEN"])
    return client.push(repo, version, nir_graph, card)
```

### 2. CLI Integration

```bash
# Publish directly from CLI
neurocuda publish cnn-nmnist-snn \
  --to-synfire \
  --org neurocuda \
  --hardware loihi2,pulsar,spinnaker2,generic \
  --version 1.0.0
```

### 3. Auto-Publish from convert()

```python
snn, stats = nc.convert(
    model, data,
    publish_to_synfire=True,     # Auto-publish after conversion
    synfire_org="neurocuda",
    hardware_targets=["loihi2", "generic"],
)
```

---

## Timeline

```
TODAY:              THIS WEEK:          NEXT WEEK:
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Publish 4    │   │ Auto-publish │   │ Synfire-     │
│ models to    │ → │ from convert │ → │ optimized    │
│ Synfire      │   │ () pipeline  │   │ NIR export   │
│ (First!)     │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
```

---

## Success Metrics

- **First organization on Synfire** with published models
- **Most published models** on the platform
- **NeuroCUDA → Synfire** becomes the standard publishing workflow
- **Every new Synfire user** discovers NeuroCUDA through the registry
- **Citation network**: Models cite NeuroCUDA → papers cite models → more users
