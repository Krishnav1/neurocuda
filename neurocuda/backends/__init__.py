"""NeuroCUDA Backends — Multi-target code generation.

Backends:
    gpu        — snnTorch CUDA-accelerated simulator (default)
    cpu        — Pure PyTorch CPU inference
    loihi      — Loihi 2 bit-accurate simulator (8-bit weights, validated vs Lava)
    spinnaker   — SpiNNaker-1 digital silicon (Manchester, NMPI queue)   [PHYSICAL]
    brainscales2 — BrainScaleS-2 analog silicon (Heidelberg, EBRAINS Lab) [PHYSICAL]
"""
from .gpu import GPUBackend
from .cpu import CPUBackend
from .loihi import LoihiBackend
from .spinnaker import SpiNNakerBackend
from .brainscales import BrainScaleS2Backend

BACKENDS = {
    # Simulators
    "gpu": GPUBackend,
    "cpu": CPUBackend,
    "loihi": LoihiBackend,
    "loihi3": LoihiBackend,
    "simulator": GPUBackend,
    # Physical silicon
    "spinnaker": SpiNNakerBackend,
    "spinnaker1": SpiNNakerBackend,
    "brainscales2": BrainScaleS2Backend,
    "brainscales": BrainScaleS2Backend,
    "bss2": BrainScaleS2Backend,
}


def get_backend(name: str):
    """Get backend by name.

    Args:
        name: Backend identifier. See BACKENDS dict for options.

    Returns:
        Backend instance ready for compile → run → benchmark.

    Examples:
        >>> backend = get_backend("spinnaker")
        >>> compiled = backend.compile(my_snn)
        >>> result = backend.run(compiled, input_rates)
        >>> info = backend.benchmark(compiled)
    """
    name_lower = name.lower()
    if name_lower not in BACKENDS:
        available = ", ".join(sorted(BACKENDS.keys()))
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
    return BACKENDS[name_lower]()


def list_backends():
    """List all available backends with types."""
    seen = set()
    result = []
    for name, cls in sorted(BACKENDS.items()):
        instance = cls()
        key = instance.name
        if key not in seen:
            seen.add(key)
            result.append({
                "name": instance.name,
                "description": instance.description,
                "is_simulator": instance.is_simulator,
                "hardware_type": getattr(instance, 'hardware_type', 'simulator'),
            })
    return result
