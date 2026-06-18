"""
NeuroCUDA — Multi-Backend Neuromorphic Compiler
=================================================
One-line API for PyTorch → Neuromorphic deployment.

Usage:
    import neurocuda as nc

    # Convert trained ANN to SNN
    snn, meta = nc.convert(model, calib_loader)

    # Fine-tune for better accuracy
    snn = nc.finetune(snn, train_loader, epochs=3)

    # Compile for target hardware
    result = nc.compile(snn, target="loihi3")
    output = result["backend"].run(result["compiled_model"], input_data)

    # Check all available targets
    print(nc.list_backends())

Targets: gpu, cpu, loihi, loihi3
"""

from .converter import convert, Calibrator, Converter
from .finetune import finetune, FineTuner
from .compiler import compile, list_backends, benchmark
from .utils import energy_estimate, fold_batchnorm, validate_snn
from . import backends

__version__ = "0.1.0"
__all__ = [
    "convert", "finetune", "compile", "list_backends", "benchmark",
    "Calibrator", "Converter", "FineTuner",
    "energy_estimate", "fold_batchnorm", "validate_snn",
    "backends",
]
