"""
NeuroCUDA → NIR (Neuromorphic Intermediate Representation) Exporter
=====================================================================
Converts NeuroCUDA SNN models to the standard NIR format for deployment
to SC-NeuroCore (FPGA), SpiNNaker (via PyNN/NIR), and other NIR-compatible
hardware backends.

NIR Spec: github.com/neuromorphs/NIR
SC-NeuroCore: pip install sc-neurocore

Pipeline:
  NeuroCUDA SNN → NIR dict → SC-NeuroCore → HLS C++ → FPGA bitstream
  NeuroCUDA SNN → NIR dict → NIRTorch → SpiNNaker (via EBRAINS)
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from ..ir import SNNGraph


def _numpy_or_scalar(value):
    """Convert torch tensor or numpy to native Python/NumPy."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        return value.tolist() if value.size < 1000 else value
    return value


def to_nir(graph_or_model, T=64, model_name="neurocuda_snn") -> Dict[str, Any]:
    """
    Convert a NeuroCUDA SNN to NIR (Neuromorphic Intermediate Representation) format.

    Parameters
    ----------
    graph_or_model : SNNGraph or torch.nn.Module
        NeuroCUDA SNN graph or a converted spiking PyTorch model.
    T : int
        Number of inference time steps.
    model_name : str
        Name for the exported model.

    Returns
    -------
    dict
        NIR-compatible dictionary with 'nodes' and 'edges'.
        Ready for import by SC-NeuroCore, NIRTorch, SpiNNaker, etc.
    """
    # If given a PyTorch model, extract IR first
    if isinstance(graph_or_model, torch.nn.Module):
        graph = SNNGraph.from_snn_model(graph_or_model, T=T)
    elif isinstance(graph_or_model, SNNGraph):
        graph = graph_or_model
    else:
        raise TypeError(f"Expected SNNGraph or nn.Module, got {type(graph_or_model)}")

    nodes: Dict[str, Dict] = {}
    edges: List[tuple] = []
    prev_node: Optional[str] = None
    node_idx = 0
    T_val = graph.metadata.get("T", T)

    for i, layer in enumerate(graph.layers):
        ltype = layer.layer_type
        params = layer.params
        name = layer.name or f"{ltype}_{i}"

        # ── Input node ──
        if i == 0 and ltype not in ("if_neuron", "avgpool", "flatten"):
            nodes["input"] = {
                "type": "Input",
                "shape": layer.input_shape or [1, 3, 32, 32],  # default CIFAR shape
            }
            prev_node = "input"

        # ── Convolution ──
        if ltype == "conv2d":
            weight = _numpy_or_scalar(params.get("weight"))
            bias = _numpy_or_scalar(params.get("bias"))
            nir_node = {
                "type": "Conv2d",
                "weight": weight,
                "stride": params.get("stride", 1),
                "padding": params.get("padding", 0),
            }
            if bias is not None:
                nir_node["bias"] = bias
            name = name or f"conv_{node_idx}"
            nodes[name] = nir_node
            if prev_node:
                edges.append((prev_node, name))
            prev_node = name
            node_idx += 1

        # ── Linear (Fully Connected) ──
        elif ltype == "linear":
            weight = _numpy_or_scalar(params.get("weight"))
            bias = _numpy_or_scalar(params.get("bias"))
            nir_node = {"type": "Linear", "weight": weight}
            if bias is not None:
                nir_node["bias"] = bias
            name = name or f"linear_{node_idx}"
            nodes[name] = nir_node
            if prev_node:
                edges.append((prev_node, name))
            prev_node = name
            node_idx += 1

        # ── IF/LIF Neuron ──
        elif ltype == "if_neuron":
            threshold = params.get("threshold", 1.0)
            nir_node = {
                "type": "LIF",
                "tau": float(T_val),  # time constant approximated by T
                "r": float(_numpy_or_scalar(threshold)),
                "v_threshold": float(_numpy_or_scalar(threshold)),
            }
            name = name or f"lif_{node_idx}"
            nodes[name] = nir_node
            if prev_node:
                edges.append((prev_node, name))
            prev_node = name
            node_idx += 1

        # ── Pooling ──
        elif ltype in ("avgpool",):
            nir_node = {
                "type": "AvgPool2d",
                "kernel_size": params.get("output_size", 1),
            }
            name = name or f"pool_{node_idx}"
            nodes[name] = nir_node
            if prev_node:
                edges.append((prev_node, name))
            prev_node = name
            node_idx += 1

        # ── Flatten (absorb into next linear, skip as separate node) ──
        elif ltype == "flatten":
            # Flatten is implicit in NIR — just shape change, skip node
            continue

        # ── Unknown ──
        else:
            print(f"[NIR Export] Warning: skipping unknown layer type '{ltype}'")

    # ── Output node ──
    if prev_node:
        nodes["output"] = {"type": "Output"}
        edges.append((prev_node, "output"))

    return {
        "name": model_name,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "T": T_val,
            "framework": "neurocuda",
            "neurocuda_version": graph.metadata.get("version", "0.2.0"),
        },
    }


def to_sc_neurocore(graph_or_model, T=64, model_name="neurocuda_snn") -> Dict[str, Any]:
    """
    Export NeuroCUDA SNN and import into SC-NeuroCore for FPGA deployment.

    Pipeline: NeuroCUDA SNN → NIR → SC-NeuroCore → HLS C++ → FPGA bitstream

    Parameters
    ----------
    graph_or_model : SNNGraph or torch.nn.Module
    T : int
        Inference time steps.
    model_name : str
        Model name.

    Returns
    -------
    dict
        {
            "nir_graph": NIR dict ready for SC-NeuroCore import,
            "hls_cpp": HLS C++ string (if SC-NeuroCore available),
            "status": "ready_for_fpga" or "sc_neurocore_missing"
        }
    """
    result = {
        "nir_graph": None,
        "hls_cpp": None,
        "status": "unknown",
    }

    # Step 1: Export to NIR
    nir_graph = to_nir(graph_or_model, T=T, model_name=model_name)
    result["nir_graph"] = nir_graph

    # Step 2: Try SC-NeuroCore import + HLS export
    try:
        from sc_neurocore.compiler.intelligence.nir_import import (
            import_nir_graph,
        )
        from sc_neurocore.compiler.intelligence.hls_export import (
            generate_hls_cpp,
        )

        imported = import_nir_graph(nir_graph, framework="neurocuda")
        hls_cpp = generate_hls_cpp(
            module_name=model_name,
            equations=imported.equations,
            data_width=16,
            fraction=8,
            hls_tool="vitis",
        )
        result["hls_cpp"] = hls_cpp
        result["equations"] = imported.equations
        result["status"] = "ready_for_fpga"
        result["next_step"] = (
            "Feed HLS C++ to Xilinx Vitis HLS → synthesize → FPGA bitstream → deploy"
        )

    except ImportError:
        result["status"] = "sc_neurocore_missing"
        result["next_step"] = (
            "pip install sc-neurocore for FPGA deployment. "
            "NIR graph is still exportable for other backends."
        )
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def to_hls_cpp(graph_or_model, T=64, model_name="neurocuda_snn") -> str:
    """
    Quick helper: Export NeuroCUDA SNN → synthesisable HLS C++ for FPGA.

    Returns HLS C++ source code string ready for Xilinx Vitis HLS.
    """
    result = to_sc_neurocore(graph_or_model, T=T, model_name=model_name)
    if result["status"] == "ready_for_fpga":
        return result["hls_cpp"]
    else:
        return f"// HLS export failed: {result.get('error', result['status'])}\n"