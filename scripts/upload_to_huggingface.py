#!/usr/bin/env python3
"""
Upload NeuroCUDA models to HuggingFace Hub.

Usage:
    python scripts/upload_to_huggingface.py --model nmnist     # Upload one model
    python scripts/upload_to_huggingface.py --all               # Upload all models
    python scripts/upload_to_huggingface.py --list              # List models ready to upload
    python scripts/upload_to_huggingface.py --card-only         # Generate model cards only

Prerequisites:
    1. pip install huggingface_hub
    2. Create HuggingFace account: https://huggingface.co/join
    3. Get API token: https://huggingface.co/settings/tokens
    4. Login: huggingface-cli login
       OR set env: export HF_TOKEN=hf_xxxxxxxx

Directory structure for each model:
    neurocuda/{model-name}/
    ├── README.md              # Model card (auto-generated)
    ├── pytorch_model.bin      # Model weights (state_dict)
    ├── config.json            # Model metadata
    └── model_info.json        # Full benchmark data
"""

import sys, os, json, argparse, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import (
    HfApi, create_repo, upload_file, upload_folder,
    login, whoami, list_models
)

from neurocuda.hub import MODEL_REGISTRY

# ===========================================================================
# Model Card Generator
# ===========================================================================

# Map our task names to valid HuggingFace pipeline tags
_HF_PIPELINE_TAG_MAP = {
    "event-camera-vision": "image-classification",
    "robotics-perception": "robotics",
    "image-classification": "image-classification",
    "digit-classification": "image-classification",
    "reinforcement-learning": "reinforcement-learning",
    "gesture-recognition": "image-classification",
    "keyword-spotting": "audio-classification",
    "anomaly-detection": "other",
}

def _get_hf_pipeline_tag(task):
    """Map our task name to a valid HuggingFace pipeline tag."""
    return _HF_PIPELINE_TAG_MAP.get(task, "other")


def generate_model_card(model_name, info):
    """Generate a HuggingFace-compatible model card in Markdown."""
    status_emoji = {"production": "✅", "beta": "⚠️", "planned": "🔮"}
    emoji = status_emoji.get(info.get("status"), "❓")
    pipeline_tag = _get_hf_pipeline_tag(info.get("task", ""))

    # Format accuracy line
    if "snn_accuracy" in info:
        accuracy_line = f"- **SNN Accuracy:** {info['snn_accuracy']}%"
        if "snn_accuracy_std" in info:
            accuracy_line += f" ± {info['snn_accuracy_std']}%"
        if "gap" in info:
            gap = info["gap"]
            direction = "BETTER than" if gap < 0 else "within"
            accuracy_line += f" | **Gap:** {gap:+.2f}% ({direction} ANN)"
        accuracy_line += "\n"
    elif "snn_solved_best" in info:
        accuracy_line = f"- **Best Solved:** {info['snn_solved_best']}\n"
        accuracy_line += f"- **Mean Solved:** {info.get('snn_solved_mean', 'N/A')}% ± {info.get('snn_solved_std', 'N/A')}%\n"
    elif "snn_solved" in info:
        accuracy_line = f"- **Solved:** {info['snn_solved']}\n"
    else:
        accuracy_line = "- **Status:** Planned — training pending\n"

    # Additional metrics
    extras = ""
    if "sparsity" in info:
        extras += f"- **Sparsity:** {info['sparsity']}%"
        if "sparsity_std" in info:
            extras += f" ± {info['sparsity_std']}%"
        extras += "\n"
    if "energy_per_inference_uj" in info:
        extras += f"- **Energy/Inference:** {info['energy_per_inference_uj']:.2f} µJ\n"
        extras += f"- **Energy vs ANN:** {info['energy_vs_ann_pct']}% reduction\n"
    if "params" in info:
        size = info.get("size_kb", info.get("size_mb", 0))
        unit = "KB" if "size_kb" in info else "MB"
        extras += f"- **Parameters:** {info['params']:,} ({size} {unit})\n"
    if "t" in info:
        extras += f"- **Timesteps:** T={info['t']}\n"

    tags_str = ", ".join(info.get("tags", []))
    hardware = info.get("hardware_validated", "GPU, CPU")

    card = f"""---
license: mit
tags:
- neurocuda
- spiking-neural-network
- snn
- neuromorphic
- {info.get("category", "other")}
{chr(10).join(f'- {t}' for t in info.get("tags", []))}
pipeline_tag: {pipeline_tag}
---

# {model_name} {emoji}

{info.get("description", "")}

## Model Details

- **Task:** {info.get("task", "N/A")}
- **Dataset:** {info.get("dataset", "N/A")}
- **Architecture:** {info.get("architecture", "N/A")}
- **Training:** {info.get("training", "N/A")}
- **Status:** {info.get("status", "unknown")}

## Performance

{accuracy_line}{extras}
## Usage

```python
import neurocuda as nc

# Load the pre-converted spiking model
snn, info = nc.hub.load("{model_name}")

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

- **Validated on:** {hardware}
- **NIR Export:** {"Yes — deployable to Loihi 2, SpiNNaker, FPGA" if info.get("nir_exportable") else "Not yet"}

## Conversion Method

{info.get("training", info.get("conversion_method", "See model card"))}

## Citation

```bibtex
@software{{neurocuda2026,
  title    = {{NeuroCUDA: A PyTorch-to-Neuromorphic Compiler}},
  author   = {{Krishna Varma}},
  year     = {{2026}},
  url      = {{https://github.com/neurocuda/neurocuda}}
}}
```

## Limitations

{f"⚠️ {info.get('note', '')}"
  if info.get("note") else
  "This is a converted spiking neural network. Accuracy was measured on the full test set with ≥3 seeds (where noted). "
  "Performance may vary on different hardware backends. See the NeuroCUDA README for detailed benchmarking methodology."}
"""
    return card


def generate_config_json(model_name, info):
    """Generate config.json for the model."""
    return {
        "model_name": model_name,
        "model_type": "neurocuda_snn",
        "neurocuda_version": "0.2.0",
        "architecture": info.get("architecture", ""),
        "task": info.get("task", ""),
        "category": info.get("category", ""),
        "t": info.get("t", 16),
        "params": info.get("params", 0),
        "status": info.get("status", "unknown"),
        "tags": info.get("tags", []),
        "conversion_method": info.get("training", info.get("conversion_method", "")),
        "nir_exportable": info.get("nir_exportable", False),
    }


# ===========================================================================
# Upload Logic
# ===========================================================================

def upload_model(model_name, token=None, dry_run=False):
    """Upload a single model to HuggingFace Hub.

    Args:
        model_name: e.g., "neurocuda/cnn-nmnist-snn"
        token: HuggingFace API token (or set HF_TOKEN env var)
        dry_run: If True, only generate files, don't upload
    """
    info = MODEL_REGISTRY.get(model_name)
    if not info:
        print(f"  ❌ Model '{model_name}' not found in registry")
        return False

    if info.get("status") == "planned":
        print(f"  🔮 {model_name} is planned — skipping upload")
        return False

    print(f"\n  Uploading: {model_name}")

    # Create output directory
    safe_name = model_name.replace("/", "_")
    out_dir = f"./scripts/hub_models/{safe_name}"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Generate model card
    card = generate_model_card(model_name, info)
    with open(f"{out_dir}/README.md", "w", encoding="utf-8") as f:
        f.write(card)
    print(f"    ✅ Model card: {out_dir}/README.md")

    # 2. Generate config
    config = generate_config_json(model_name, info)
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"    ✅ Config: {out_dir}/config.json")

    # 3. Save full model info
    with open(f"{out_dir}/model_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"    ✅ Info: {out_dir}/model_info.json")

    # 4. Check for weights
    weight_path = _find_weights(model_name)
    if weight_path and os.path.exists(weight_path):
        import shutil
        dest = f"{out_dir}/pytorch_model.bin"
        shutil.copy2(weight_path, dest)
        size_kb = os.path.getsize(dest) / 1024
        print(f"    ✅ Weights: {dest} ({size_kb:.0f} KB)")
    else:
        print(f"    ⚠️  No weights found — model card + config only")
        print(f"       Generate weights: python scripts/export_hub_models.py")

    if dry_run:
        print(f"    [DRY RUN] Files ready in {out_dir}/")
        return True

    # 5. Upload to HuggingFace
    try:
        api = HfApi(token=token)

        # Create repo if it doesn't exist
        repo_id = model_name
        try:
            create_repo(repo_id, token=token, exist_ok=True)
            print(f"    ✅ Repo: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"    ⚠️  Repo creation: {e}")

        # Upload files
        for filename in ["README.md", "config.json", "model_info.json", "pytorch_model.bin"]:
            filepath = f"{out_dir}/{filename}"
            if os.path.exists(filepath):
                upload_file(
                    path_or_fileobj=filepath,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    token=token,
                )
        print(f"    🚀 Uploaded to https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"    ❌ Upload failed: {e}")
        return False


def _find_weights(model_name):
    """Find model weights in local checkpoints."""
    mapping = {
        "neurocuda/cnn-nmnist-snn": "./checkpoints/hub/nmnist_cnn_snn.pt",
        "neurocuda/robotics-perception-snn": "./checkpoints/hub/robotics_perception_snn.pt",
        "neurocuda/resnet18-cifar10-snn": "./checkpoints/hub/resnet18_cifar10_snn.pt",
        "neurocuda/strongcnn-cifar10-snn": "./checkpoints/hub/strongcnn_cifar10_snn.pt",
        "neurocuda/sew-resnet-cifar10-snn": "./checkpoints/hub/sew_resnet_cifar10_snn.pt",
        "neurocuda/mlp-mnist-snn": "./checkpoints/hub/mlp_mnist_snn.pt",
        "neurocuda/dqn-cartpole-snn": "./checkpoints/hub/cartpole_dqn_snn.pt",
        "neurocuda/lif-dqn-cartpole-snn": "./checkpoints/hub/cartpole_lif_dqn_snn.pt",
    }
    return mapping.get(model_name, f"./checkpoints/hub/{model_name.replace('/', '_')}.pt")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Upload NeuroCUDA models to HuggingFace Hub")
    parser.add_argument("--model", type=str, help="Upload specific model")
    parser.add_argument("--all", action="store_true", help="Upload all non-planned models")
    parser.add_argument("--card-only", action="store_true", help="Generate model cards only (no upload)")
    parser.add_argument("--list", action="store_true", help="List uploadable models")
    parser.add_argument("--token", type=str, help="HuggingFace API token")
    parser.add_argument("--dry-run", action="store_true", help="Generate files without uploading")
    args = parser.parse_args()

    # Get token
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if args.list:
        print("\nModels ready for HuggingFace upload:\n")
        for name, info in MODEL_REGISTRY.items():
            status = info.get("status", "unknown")
            icon = {"production": "✅", "beta": "⚠️", "planned": "🔮"}.get(status, "❓")
            has_weights = os.path.exists(_find_weights(name))
            weight_status = "📦 weights ready" if has_weights else "⬜ needs weights"
            print(f"  {icon} {name}")
            print(f"     Status: {status} | Category: {info.get('category', '?')}")
            print(f"     Task: {info.get('task', '?')} | {weight_status}")
            print()
        return

    models_to_upload = []

    if args.model:
        if args.model in MODEL_REGISTRY:
            models_to_upload.append(args.model)
        else:
            # Try partial match
            matches = [n for n in MODEL_REGISTRY if args.model in n]
            if matches:
                models_to_upload = matches
            else:
                print(f"Model '{args.model}' not found. Use --list to see available models.")
                return
    elif args.all:
        models_to_upload = [n for n, i in MODEL_REGISTRY.items()
                           if i.get("status") != "planned"]
    else:
        print("Specify --model, --all, or --list")
        return

    print("=" * 60)
    print("  NeuroCUDA → HuggingFace Hub Upload")
    print(f"  Models: {len(models_to_upload)}")
    if not token:
        print("  ⚠️  No HF token set. Use --token or set HF_TOKEN env var.")
        print("  Generate token: https://huggingface.co/settings/tokens")
        if args.card_only or args.dry_run:
            print("  Continuing with --card-only / --dry-run mode...")
        else:
            print("  Run with --card-only to generate model cards without uploading.")
            return
    print("=" * 60)

    success = 0
    for model_name in sorted(models_to_upload):
        if upload_model(model_name, token=token, dry_run=args.dry_run or args.card_only):
            success += 1

    print(f"\n{'=' * 60}")
    print(f"  Done: {success}/{len(models_to_upload)} models processed")
    if args.card_only or args.dry_run:
        print(f"  Model cards in: scripts/hub_models/")
    else:
        print(f"  View at: https://huggingface.co/neurocuda")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
