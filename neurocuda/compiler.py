"""
NeuroCUDA Compiler — Unified Multi-Backend Deployment
======================================================
One API. Any model. Any chip.

neurocuda.compile(model, target="loihi3")
neurocuda.compile(model, target="gpu")
neurocuda.compile(model, target="cpu")
"""
import torch
from typing import Any, Dict, Optional
from .backends import get_backend, BACKENDS


def compile(
    model: torch.nn.Module,
    target: str = "gpu",
    T: int = 64,
    calibrate: bool = True,
    calib_loader=None,
    percentile: float = 95.0,
    validate: bool = False,
    test_loader=None,
) -> Dict[str, Any]:
    """
    Compile a converted SNN model for a target hardware backend.

    Args:
        model: Converted SNN model (from neurocuda.convert())
        target: Backend name — "gpu", "cpu", "loihi", "loihi3"
        T: Inference timesteps (default 64)
        calibrate: Quantize weights for target hardware
        calib_loader: DataLoader for calibration (optional)
        percentile: Percentile for threshold calibration
        validate: Run accuracy validation after compilation
        test_loader: DataLoader for validation

    Returns:
        Dict with:
          - compiled_model: The compiled SNN ready for inference
          - backend: Backend instance (with .run(), .benchmark())
          - metadata: Dict with target, T, energy estimate, accuracy

    Example:
        result = nc.compile(snn, target="loihi3")
        output = result["backend"].run(result["compiled_model"], input_data)
    """
    # Get backend
    backend = get_backend(target)

    # Compile model for target hardware
    compiled = backend.compile(model)

    # Energy estimation
    energy = None
    if hasattr(backend, "estimate_energy"):
        energy = backend.estimate_energy(compiled, T=T)

    # Optional validation
    accuracy = None
    if validate and test_loader is not None:
        compiled.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target_labels in test_loader:
                data = data.to(next(compiled.parameters()).device)
                target_labels = target_labels.to(data.device)
                out = compiled(data)
                correct += out.max(1)[1].eq(target_labels).sum().item()
                total += target_labels.size(0)
        accuracy = 100.0 * correct / total

    return {
        "compiled_model": compiled,
        "backend": backend,
        "metadata": {
            "target": target,
            "backend_description": backend.description,
            "T": T,
            "is_simulator": backend.is_simulator,
            "energy": energy,
            "accuracy": accuracy,
        },
    }


def list_backends() -> Dict[str, str]:
    """List all available compilation targets."""
    return {name: cls().description for name, cls in BACKENDS.items()}


def benchmark(model, target: str = "gpu", input_shape=(1, 3, 32, 32), T=64, iterations=100):
    """Benchmark a compiled model on a target backend."""
    backend = get_backend(target)
    compiled = backend.compile(model)
    return backend.benchmark(compiled, input_shape, T, iterations)
