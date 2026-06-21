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

from .converter import convert, Calibrator, measure_sparsity
from .finetune import finetune, FineTuner
from .compiler import compile, list_backends, benchmark
from .utils import energy_estimate, fold_batchnorm, validate_snn
from .export.nir_exporter import to_nir, to_sc_neurocore, to_hls_cpp
from . import backends
from . import hub

__version__ = "0.2.0"
__all__ = [
    "convert", "finetune", "compile", "list_backends", "benchmark",
    "to_nir", "to_sc_neurocore", "to_hls_cpp",
    "Calibrator", "FineTuner", "measure_sparsity",
    "energy_estimate", "fold_batchnorm", "validate_snn",
    "backends",
]
