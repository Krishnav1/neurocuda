"""NeuroCUDA Backends — Multi-target code generation."""
from .gpu import GPUBackend
from .cpu import CPUBackend
from .loihi import LoihiBackend

BACKENDS = {
    "gpu": GPUBackend,
    "cpu": CPUBackend,
    "loihi": LoihiBackend,
    "loihi3": LoihiBackend,
    "simulator": GPUBackend,
}

def get_backend(name: str):
    """Get backend by name."""
    if name not in BACKENDS:
        available = ", ".join(BACKENDS.keys())
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
    return BACKENDS[name]()
