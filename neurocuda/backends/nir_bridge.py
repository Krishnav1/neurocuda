"""
NeuroCUDA → official NIR (neuromorphs/NIR) graph builder.

Converts PyTorch SNN modules (Linear + IFNeuron/LIFNeuron) into nir.NIRGraph
for Lava import and cross-platform exchange.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import nir
except ImportError:  # pragma: no cover - optional dep
    nir = None  # type: ignore

try:
    from models import IFNeuron, LIFNeuron
except ImportError:
    from ..models import IFNeuron, LIFNeuron  # type: ignore


def nir_available() -> bool:
    return nir is not None


def _as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def _if_threshold(module: nn.Module, fan_in: int) -> np.ndarray:
    """Per-neuron thresholds for NIR (shape must match preceding Affine output)."""
    thr = module.thresh.detach().cpu().float()
    size = max(int(fan_in), 1)
    if thr.ndim == 0:
        return np.full(size, float(thr.item()), dtype=np.float32)
    if thr.numel() == 1:
        return np.full(size, float(thr.item()), dtype=np.float32)
    if thr.numel() != size:
        raise ValueError(
            f"IF threshold size {thr.numel()} != layer width {size}"
        )
    return thr.numpy().astype(np.float32)


def _extract_ordered_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return compute layers in forward order (skip Flatten/Dropout)."""
    ordered: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, (nn.Linear, IFNeuron, LIFNeuron)):
            ordered.append((name, module))
    return ordered


def snn_to_nir_graph(
    model: nn.Module,
    T: int = 32,
    type_check: bool = True,
) -> Any:
    """
    Build an official nir.NIRGraph from a NeuroCUDA SNN (Linear + IF/LIF).

    Supports feed-forward MLP-style models. Conv2d chains require manual
    NIR export via neurocuda.to_nir() until Conv NIR nodes are wired here.
    """
    if nir is None:
        raise ImportError(
            "nir package required. Install with: pip install neurocuda[nir]"
        )

    layers = _extract_ordered_layers(model)
    if not layers:
        raise ValueError("No Linear/IFNeuron layers found in model.")

    nir_nodes: List[Any] = []
    param_audit: List[Dict[str, Any]] = []
    fan_in = 1

    for name, module in layers:
        if isinstance(module, nn.Linear):
            w = _as_numpy(module.weight)
            b = _as_numpy(module.bias) if module.bias is not None else np.zeros(
                module.out_features, dtype=np.float32
            )
            node = nir.Affine(weight=w, bias=b)
            nir_nodes.append(node)
            fan_in = int(module.out_features)
            param_audit.append(
                {
                    "layer": name,
                    "type": "Affine",
                    "out_features": fan_in,
                    "in_features": int(module.in_features),
                }
            )
        elif isinstance(module, (IFNeuron, LIFNeuron)):
            thr = _if_threshold(module, fan_in)
            if isinstance(module, LIFNeuron):
                tau = np.ones_like(thr, dtype=np.float32)
                v_leak = np.zeros_like(thr, dtype=np.float32)
                node = nir.LIF(
                    tau=tau,
                    r=thr,
                    v_leak=v_leak,
                    v_threshold=thr,
                )
                ntype = "LIF"
            else:
                node = nir.IF(r=thr, v_threshold=thr)
                ntype = "IF"
            nir_nodes.append(node)
            param_audit.append(
                {
                    "layer": name,
                    "type": ntype,
                    "threshold_min": float(thr.min()),
                    "threshold_max": float(thr.max()),
                    "threshold_mean": float(thr.mean()),
                }
            )
        else:
            raise TypeError(f"Unsupported module in NIR bridge: {type(module)}")

    graph = nir.NIRGraph.from_list(*nir_nodes, type_check=type_check)
    graph.metadata["T"] = int(T)
    graph.metadata["framework"] = "neurocuda"
    graph.metadata["param_audit"] = param_audit
    return graph


def audit_nir_params(graph: Any) -> Dict[str, Any]:
    """Summarize neuron parameters in a NIR graph for Lava fixed_pt debugging."""
    if nir is None:
        raise ImportError("nir package required")

    summary: Dict[str, Any] = {"nodes": []}
    for name, node in graph.nodes.items():
        entry: Dict[str, Any] = {"name": name, "type": node.__class__.__name__}
        if hasattr(node, "v_threshold"):
            vt = np.asarray(node.v_threshold)
            entry["v_threshold"] = {
                "shape": list(vt.shape),
                "min": float(vt.min()),
                "max": float(vt.max()),
                "mean": float(vt.mean()),
            }
        if hasattr(node, "r"):
            r = np.asarray(node.r)
            entry["r"] = {
                "shape": list(r.shape),
                "min": float(r.min()),
                "max": float(r.max()),
            }
        if hasattr(node, "weight"):
            w = np.asarray(node.weight)
            entry["weight_shape"] = list(w.shape)
        summary["nodes"].append(entry)
    return summary
