"""
NeuroCUDA — Multi-Backend Neuromorphic Compiler.

Usage:
    import neurocuda as nc
    snn = nc.convert(model, calib_loader)
    snn = nc.finetune(snn, train_loader)
"""
from .converter import convert, Calibrator, Converter
from .finetune import finetune, FineTuner
from .utils import energy_estimate, fold_batchnorm

__version__ = "0.1.0"
__all__ = ["convert", "finetune", "Calibrator", "Converter",
           "FineTuner", "energy_estimate", "fold_batchnorm"]
