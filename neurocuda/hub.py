"""
NeuroCUDA Model Hub — Pre-converted, pre-validated spiking neural networks.

Usage:
    import neurocuda as nc

    # List available models
    print(nc.hub.list())

    # Load a pre-converted SNN
    snn, info = nc.hub.load("neurocuda/cnn-nmnist-snn")
    snn.eval()
    output = snn(event_camera_data)  # Binary spikes, 92% sparse

    # Search models
    models = nc.hub.search("vision")

    # Get model info without downloading
    info = nc.hub.info("neurocuda/resnet18-cifar10-snn")
"""

import os
import json
import torch
import torch.nn as nn

# ===========================================================================
# Model Registry
# ===========================================================================

MODEL_REGISTRY = {
    # =========================================================================
    # VISION — Event Camera (Neuromorphic Vision Sensors)
    # =========================================================================
    "neurocuda/cnn-nmnist-snn": {
        "task": "event-camera-vision",
        "category": "vision",
        "dataset": "N-MNIST",
        "architecture": "3-layer CNN (2 input channels, 34×34)",
        "training": "ANN → CS-QCFS → IF + BPTT FT (conversion)",
        "ann_accuracy": 99.70,
        "snn_accuracy": 99.88,
        "snn_accuracy_std": 0.02,
        "gap": -0.18,
        "sparsity": 91.7,
        "sparsity_std": 0.5,
        "params": 147466,
        "size_kb": 576,
        "num_seeds": 3,
        "t": 16,
        "hardware_validated": "GPU, CPU, Loihi 2 simulator",
        "nir_exportable": True,
        "status": "production",
        "tags": ["vision", "event-camera", "n-mnist", "classification", "robotics", "flagship"],
        "description": "Event-camera object classification SNN. Beats the original ANN by 0.18%. "
                       "92% sparse — only 8% of neurons fire. Best-in-class conversion. "
                       "Our flagship model.",
    },
    "neurocuda/robotics-perception-snn": {
        "task": "robotics-perception",
        "category": "vision",
        "dataset": "N-MNIST (event camera)",
        "architecture": "3-layer CNN (2 input channels, 34×34), 5D-native",
        "training": "ANN → CS-QCFS → IF + BPTT FT (conversion, 5 epochs, 20K data)",
        "ann_accuracy": 99.70,
        "snn_accuracy": 99.95,
        "gap": -0.25,
        "sparsity": 92.06,
        "energy_per_inference_uj": 13.02,
        "energy_vs_ann_pct": 49,
        "params": 147466,
        "size_kb": 576,
        "t": 16,
        "hardware_validated": "GPU, CPU, Loihi 2 simulator, NIR-exported",
        "nir_exportable": True,
        "status": "production",
        "tags": ["vision", "robotics", "event-camera", "n-mnist", "perception", "energy"],
        "description": "Full robotics perception pipeline — event camera → SNN → deploy. "
                       "99.95% accuracy (beats ANN). 92% sparse. 49% energy reduction vs ANN. "
                       "NIR-exported and ready for Loihi 2 / SpiNNaker / FPGA deployment.",
    },
    # =========================================================================
    # VISION — Standard Image Classification
    # =========================================================================
    "neurocuda/resnet18-cifar10-snn": {
        "task": "image-classification",
        "category": "vision",
        "dataset": "CIFAR-10",
        "architecture": "ResNet-18 (CIFAR variant, 8 residual blocks, skip connections)",
        "training": "ANN → QCFS → IF (direct conversion, no fine-tune). T=32.",
        "ann_accuracy": 95.56,
        "ann_accuracy_std": 0.11,
        "snn_accuracy": 94.61,
        "snn_accuracy_std": 0.14,
        "gap": 0.95,
        "sparsity": 93.7,
        "params": 11173962,
        "size_mb": 42.6,
        "num_seeds": 3,
        "t": 32,
        "hardware_validated": "GPU, CPU, Loihi 2 simulator",
        "nir_exportable": True,
        "status": "production",
        "tags": ["vision", "cifar-10", "resnet", "classification", "deep", "residual"],
        "description": "Deep residual SNN for CIFAR-10 classification. Gap of 0.95% at T=32. "
                       "Uses direct QCFS→IF replacement without fine-tuning (standard for deep ResNets). "
                       "Skip connections handled by Kahn's topology sort in NIR executor.",
    },
    "neurocuda/strongcnn-cifar10-snn": {
        "task": "image-classification",
        "category": "vision",
        "dataset": "CIFAR-10",
        "architecture": "StrongCNN (7-layer, BatchNorm, wider channels)",
        "training": "ANN → QCFS → IF + BPTT FT (conversion + fine-tune)",
        "ann_accuracy": 80.4,
        "snn_accuracy": 74.3,
        "gap": 6.0,
        "params": 4800000,
        "size_mb": 18.3,
        "num_seeds": 1,
        "t": 16,
        "hardware_validated": "GPU, CPU",
        "nir_exportable": True,
        "status": "beta",
        "tags": ["vision", "cifar-10", "strongcnn", "classification", "medium"],
        "description": "7-layer CNN SNN for CIFAR-10. Gap of 6% with conversion + fine-tuning. "
                       "Larger gap than ResNet-18 due to higher non-linearity. "
                       "Good for studying conversion challenges in medium-depth networks.",
        "note": "Single seed result. Multi-seed verification pending.",
    },
    "neurocuda/mlp-mnist-snn": {
        "task": "digit-classification",
        "category": "vision",
        "dataset": "MNIST",
        "architecture": "3-layer MLP (784→256→256→10)",
        "training": "ANN → QCFS → IF (direct conversion, no FT)",
        "ann_accuracy": 97.8,
        "snn_accuracy": 97.4,
        "gap": 0.4,
        "params": 269322,
        "size_kb": 1052,
        "t": 16,
        "hardware_validated": "GPU, CPU",
        "nir_exportable": True,
        "status": "production",
        "tags": ["vision", "digit", "mnist", "classification", "quickstart", "beginner"],
        "description": "Simple MLP SNN for MNIST digit classification. "
                       "The classic beginner example — great for learning the conversion pipeline. "
                       "Converts in under 1 minute on CPU.",
    },
    # =========================================================================
    # CONTROL — Reinforcement Learning
    # =========================================================================
    "neurocuda/dqn-cartpole-snn": {
        "task": "reinforcement-learning",
        "category": "control",
        "dataset": "CartPole-v1",
        "architecture": "3-layer MLP DQN (4→128→128→2) with LIF neurons",
        "training": "ANN weight transfer + BPTT FT (conversion, surrogate gradient)",
        "ann_solved": "100% (early-stop at Train100≥195)",
        "snn_solved_best": "100%",
        "snn_solved_mean": 19.0,
        "snn_solved_std": 26.0,
        "sparsity": 74.5,
        "sparsity_std": 2.1,
        "params": 17922,
        "size_kb": 70,
        "num_seeds": 7,
        "t": 16,
        "hardware_validated": "GPU, CPU",
        "nir_exportable": False,
        "status": "beta",
        "tags": ["control", "reinforcement-learning", "cartpole", "dqn", "rl", "conversion"],
        "description": "DQN policy network converted to LIF SNN for CartPole control. "
                       "Best seed reaches 100% solved. ~29% seed success rate — DQN training "
                       "produces policies with varying robustness to ReLU→LIF transfer. "
                       "Use direct LIF training for 100% reliability.",
        "note": "Stochastic benchmark. See neurocuda/lif-dqn-cartpole-snn for the reliable version.",
    },
    "neurocuda/lif-dqn-cartpole-snn": {
        "task": "reinforcement-learning",
        "category": "control",
        "dataset": "CartPole-v1",
        "architecture": "3-layer MLP DQN (4→128→128→2) with LIF neurons",
        "training": "Direct LIF SNN training from scratch (BPTT + surrogate gradient)",
        "snn_solved": "100% (solved at ~350 episodes)",
        "sparsity": 68.5,
        "params": 17922,
        "size_kb": 70,
        "t": 16,
        "hardware_validated": "GPU, CPU",
        "nir_exportable": False,
        "status": "production",
        "tags": ["control", "reinforcement-learning", "cartpole", "dqn", "rl", "direct-training"],
        "description": "LIF SNN DQN trained from scratch via BPTT with surrogate gradients. "
                       "100% reliable — always reaches 100% solved. The recommended approach for "
                       "spiking reinforcement learning. No conversion needed — native SNN training.",
    },
    "neurocuda/sew-resnet-cifar10-snn": {
        "task": "image-classification",
        "category": "vision",
        "dataset": "CIFAR-10",
        "architecture": "SEW-ResNet (Spiking Element-Wise ResNet, 18 layers)",
        "training": "Direct SNN training from scratch (BPTT, surrogate gradient, 50 epochs)",
        "snn_accuracy": 67.7,
        "params": 11170000,
        "size_mb": 42.6,
        "num_seeds": 1,
        "t": 8,
        "hardware_validated": "GPU, CPU",
        "nir_exportable": True,
        "status": "beta",
        "tags": ["vision", "cifar-10", "sew-resnet", "classification", "direct-training", "deep"],
        "description": "SEW-ResNet trained directly as an SNN from scratch (no ANN→SNN conversion). "
                       "67.7% at T=8 timesteps. Demonstrates direct SNN training with surrogate "
                       "gradients. Accuracy improves with more timesteps and training epochs.",
        "note": "Single seed, 50 epochs. Extended training (200+ epochs) expected to reach 85%+.",
    },
    # =========================================================================
    # COMING SOON — Planned Models
    # =========================================================================
    "neurocuda/gesture-dvs-snn": {
        "task": "gesture-recognition",
        "category": "vision",
        "dataset": "DVS128-Gesture",
        "architecture": "5-layer CNN + LIF (3D convolutions for temporal)",
        "training": "Direct SNN training (BPTT) or ANN→SNN conversion",
        "params": 500000,
        "size_kb": 2000,
        "t": 16,
        "hardware_validated": "Planned: GPU, CPU, Loihi 2 simulator",
        "nir_exportable": True,
        "status": "planned",
        "tags": ["vision", "gesture", "dvs128", "event-camera", "classification"],
        "description": "Hand gesture recognition from DVS event camera data. "
                       "11 gesture classes, 122 subjects. Key application for AR/VR "
                       "and human-robot interaction. Coming Q3 2026.",
    },
    "neurocuda/keyword-spotting-snn": {
        "task": "keyword-spotting",
        "category": "audio",
        "dataset": "Google Speech Commands (GSC v2)",
        "architecture": "6-layer CNN + LIF (1D convolutions over audio)",
        "training": "ANN→SNN conversion or direct BPTT training",
        "params": 250000,
        "size_kb": 1000,
        "t": 16,
        "hardware_validated": "Planned: GPU, CPU, Loihi 2 simulator",
        "nir_exportable": True,
        "status": "planned",
        "tags": ["audio", "keyword-spotting", "gsc", "speech", "edge", "always-on"],
        "description": "Always-on keyword spotting with spiking networks. "
                       "35 keywords. Target: <1mW on neuromorphic hardware. "
                       "Key commercial use case for edge AI. Coming Q3 2026.",
    },
    "neurocuda/anomaly-detection-snn": {
        "task": "anomaly-detection",
        "category": "industrial",
        "dataset": "MVTec AD (industrial anomaly detection)",
        "architecture": "Spiking autoencoder (Conv-IF encoder + Conv decoder)",
        "training": "ANN autoencoder → QCFS → IF conversion",
        "params": 2000000,
        "size_mb": 8,
        "t": 16,
        "hardware_validated": "Planned: GPU, CPU, Loihi 2 simulator",
        "nir_exportable": True,
        "status": "planned",
        "tags": ["industrial", "anomaly-detection", "autoencoder", "quality-control", "manufacturing"],
        "description": "Spiking autoencoder for industrial anomaly detection. "
                       "Detect defects in manufacturing without frame-based cameras. "
                       "Continuous event-driven monitoring for Industry 4.0. Coming Q4 2026.",
    },
}

# HuggingFace namespace — where models are actually hosted
_HF_NAMESPACE = "Krishnav1234"


# ===========================================================================
# Public API
# ===========================================================================

def list(category=None, status=None):
    """List available models in the NeuroCUDA hub.

    Args:
        category: Filter by category ('vision', 'control', 'audio', 'industrial')
        status: Filter by status ('production', 'beta', 'planned')

    Returns:
        list of dicts: Each with name, task, accuracy, size, and tags.
    """
    models = []
    for name, info in MODEL_REGISTRY.items():
        if category and info.get("category") != category:
            continue
        if status and info.get("status") != status:
            continue
        models.append({
            "name": name,
            "task": info["task"],
            "category": info.get("category", "other"),
            "status": info.get("status", "unknown"),
            "snn_accuracy": info.get("snn_accuracy", info.get("snn_solved_best",
                                info.get("snn_solved", "planned"))),
            "gap": info.get("gap", None),
            "params": info["params"],
            "size": info.get("size_kb", info.get("size_mb", 0)),
            "tags": info["tags"],
        })
    return models


def categories():
    """List model categories."""
    cats = set()
    for info in MODEL_REGISTRY.values():
        if "category" in info:
            cats.add(info["category"])
    return sorted(cats)


def info(model_name):
    """Get detailed information about a model without downloading it.

    Args:
        model_name: e.g., "neurocuda/cnn-nmnist-snn"

    Returns:
        dict: Full model metadata.
    """
    if model_name not in MODEL_REGISTRY:
        available = "\n  ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available:\n  {available}")
    return MODEL_REGISTRY[model_name].copy()


def search(query):
    """Search models by tag or keyword.

    Args:
        query: Search term (e.g., "vision", "robotics", "cifar")

    Returns:
        list of dicts: Matching models with name and key info.
    """
    results = []
    query_lower = query.lower()
    for name, meta in MODEL_REGISTRY.items():
        # Search in name, task, tags, description
        searchable = name.lower() + " " + meta["task"].lower() + " " + \
                     " ".join(meta.get("tags", [])).lower() + " " + \
                     meta.get("description", "").lower()
        if query_lower in searchable:
            results.append({
                "name": name,
                "task": meta["task"],
                "snn_accuracy": meta.get("snn_accuracy", meta.get("snn_solved_best", "N/A")),
                "description": meta["description"],
            })
    return results


def load(model_name, device=None, use_hf=True):
    """Load a pre-converted spiking neural network.

    Tries local checkpoints first, then falls back to HuggingFace Hub.

    Args:
        model_name: e.g., "neurocuda/cnn-nmnist-snn"
        device: torch device (auto-detected if None)
        use_hf: If True, try downloading from HuggingFace when local weights missing

    Returns:
        (model, info_dict): The loaded SNN and its metadata.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_info = info(model_name)

    if model_info.get("status") == "planned":
        raise ValueError(
            f"Model '{model_name}' is planned but not yet trained.\n"
            f"Status: {model_info.get('status')}\n"
            f"Expected: {model_info.get('description', '')}"
        )

    # Build the model architecture based on the model name
    model = _build_model(model_name, device)

    # Try loading weights: local → HuggingFace → fail gracefully
    weight_path = _get_weights_path(model_name)
    weights_loaded = False

    # 1. Try local checkpoint
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        weights_loaded = True
        model_info["weights_source"] = "local"

    # 2. Try HuggingFace Hub
    elif use_hf:
        try:
            from huggingface_hub import hf_hub_download
            # Map model name to HF repo: neurocuda/cnn-nmnist-snn → Krishnav1234/neurocuda-cnn-nmnist-snn
            hf_model_name = model_name.replace('/', '-')
            hf_repo_id = f"{_HF_NAMESPACE}/{hf_model_name}"
            # Use token if available
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            hf_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename="pytorch_model.bin",
                token=hf_token,
                cache_dir=os.path.join(os.path.dirname(weight_path), ".hf_cache"),
            )
            state_dict = torch.load(hf_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            weights_loaded = True
            model_info["weights_source"] = "huggingface"
            # Cache locally for next time
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            torch.save(state_dict, weight_path)
        except ImportError:
            print("  Tip: pip install huggingface_hub to download models from HuggingFace")
        except Exception as e:
            print(f"  HuggingFace download failed: {e}")

    # 3. No weights available
    if not weights_loaded:
        model_info["weights_loaded"] = False
        model_info["weights_path"] = weight_path
        print(f"  Note: No weights found for {model_name}")
        print(f"  Local path: {weight_path}")
        print(f"  HuggingFace: https://huggingface.co/{model_name}")
        print(f"  Generate: python scripts/export_hub_models.py")
    else:
        model_info["weights_loaded"] = True

    model = model.to(device).eval()
    return model, model_info


# ===========================================================================
# Internal helpers
# ===========================================================================

def _build_model(model_name, device):
    """Build the correct model architecture for a given hub model."""
    from models import IFNeuron, LIFNeuron

    # --- Event Camera / NMNIST models ---
    if "nmnist" in model_name or "robotics" in model_name:
        class NMNISTCNNSNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(2, 32, 5, padding=2, bias=False)
                self.bn1 = nn.BatchNorm2d(32)
                self.if1 = IFNeuron(thresh=2.0, alpha=2.0, num_channels=32)
                self.pool1 = nn.AvgPool2d(2)
                self.conv2 = nn.Conv2d(32, 64, 5, padding=2, bias=False)
                self.bn2 = nn.BatchNorm2d(64)
                self.if2 = IFNeuron(thresh=2.0, alpha=2.0, num_channels=64)
                self.pool2 = nn.AvgPool2d(2)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
                self.bn3 = nn.BatchNorm2d(128)
                self.if3 = IFNeuron(thresh=2.0, alpha=2.0, num_channels=128)
                self.pool3 = nn.AvgPool2d(2)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(2048, 10)

            def forward(self, x):
                B = T = None
                if x.dim() == 5:
                    B, T, C, H, W = x.shape
                    x = x.reshape(B * T, C, H, W)
                x = self.pool1(self.if1(self.bn1(self.conv1(x))))
                x = self.pool2(self.if2(self.bn2(self.conv2(x))))
                x = self.pool3(self.if3(self.bn3(self.conv3(x))))
                x = self.flatten(x)
                x = self.fc(x)
                if B is not None:
                    x = x.reshape(B, T, -1).mean(dim=1)
                return x

        return NMNISTCNNSNN().to(device)

    # --- MNIST MLP ---
    elif "mnist" in model_name:
        class MLPMNISTSNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(784, 256)
                self.if1 = IFNeuron(thresh=1.0, alpha=2.0)
                self.fc2 = nn.Linear(256, 256)
                self.if2 = IFNeuron(thresh=1.0, alpha=2.0)
                self.fc3 = nn.Linear(256, 10)

            def forward(self, x):
                B = T = None
                if x.dim() == 5:
                    B, T, C, H, W = x.shape
                    x = x.reshape(B * T, C, H, W)
                x = self.flatten(x)
                x = self.if1(self.fc1(x))
                x = self.if2(self.fc2(x))
                x = self.fc3(x)
                if B is not None:
                    x = x.reshape(B, T, -1).mean(dim=1)
                return x

        return MLPMNISTSNN().to(device)

    # --- CartPole DQN (works for both conversion and direct training versions) ---
    elif "cartpole" in model_name:
        class CartpoleLIFDQN(nn.Module):
            def __init__(self, T=16):
                super().__init__()
                self.T = T
                self.fc1 = nn.Linear(4, 128)
                self.lif1 = LIFNeuron(threshold=1.0, beta=0.5, alpha=2.0)
                self.fc2 = nn.Linear(128, 128)
                self.lif2 = LIFNeuron(threshold=1.0, beta=0.5, alpha=2.0)
                self.fc3 = nn.Linear(128, 2)

            def forward(self, x):
                B = x.size(0)
                q = torch.zeros(B, 2, device=x.device)
                for t in range(self.T):
                    h = self.lif1(self.fc1(x))
                    h = self.lif2(self.fc2(h))
                    q = q + self.fc3(h)
                return q / self.T

        return CartpoleLIFDQN().to(device)

    # --- CIFAR-10 ResNet/SEW-ResNet/StrongCNN ---
    elif any(x in model_name for x in ["resnet", "cifar10", "strongcnn", "sew"]):
        from models import resnet18_cifar
        print(f"  Note: {model_name} uses ResNet-18 architecture from models.py")
        print(f"  Full SNN weights require the conversion pipeline.")
        print(f"  Run: python gate3_qcfs_convert.py --seed 0")
        raise NotImplementedError(
            f"{model_name} requires the conversion pipeline.\n"
            f"Run: python gate3_qcfs_convert.py --seed 0\n"
            f"Or: python scripts/export_hub_models.py --model cifar10"
        )

    else:
        raise ValueError(f"Unknown model architecture for: {model_name}")


def _get_weights_path(model_name):
    """Get the expected checkpoint path for a hub model."""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "hub")

    mapping = {
        "neurocuda/cnn-nmnist-snn": "nmnist_cnn_snn.pt",
        "neurocuda/robotics-perception-snn": "robotics_perception_snn.pt",
        "neurocuda/resnet18-cifar10-snn": "resnet18_cifar10_snn.pt",
        "neurocuda/strongcnn-cifar10-snn": "strongcnn_cifar10_snn.pt",
        "neurocuda/sew-resnet-cifar10-snn": "sew_resnet_cifar10_snn.pt",
        "neurocuda/mlp-mnist-snn": "mlp_mnist_snn.pt",
        "neurocuda/dqn-cartpole-snn": "cartpole_dqn_snn.pt",
        "neurocuda/lif-dqn-cartpole-snn": "cartpole_lif_dqn_snn.pt",
    }

    filename = mapping.get(model_name, f"{model_name.replace('/', '_')}.pt")
    return os.path.join(base_dir, filename)
