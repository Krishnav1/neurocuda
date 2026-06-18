"""
NeuroCUDA Intermediate Representation (IR)
===========================================
Hardware-agnostic SNN graph format. NIR-compatible.

Each SNN is represented as an ordered list of layers:
  Conv2D → IFNeuron → Conv2D → IFNeuron → ... → Linear → Output
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import torch
import json


@dataclass
class SNNLayer:
    """One layer in the SNN graph."""
    layer_type: str  # "conv2d", "linear", "if_neuron", "avgpool", "flatten", "input", "output"
    params: Dict[str, Any] = field(default_factory=dict)
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    name: str = ""


class SNNGraph:
    """Hardware-agnostic SNN representation."""

    def __init__(self, name="snn_model"):
        self.name = name
        self.layers: List[SNNLayer] = []
        self.metadata: Dict[str, Any] = {
            "T": 64,
            "framework": "neurocuda",
            "version": "0.1.0",
        }

    def add(self, layer: SNNLayer):
        self.layers.append(layer)
        return self

    def add_conv2d(self, weight, bias=None, stride=1, padding=0, name=""):
        self.add(SNNLayer("conv2d", {
            "weight": weight.clone(),
            "bias": bias.clone() if bias is not None else None,
            "stride": stride,
            "padding": padding,
        }, name=name))
        return self

    def add_if_neuron(self, threshold=1.0, name=""):
        self.add(SNNLayer("if_neuron", {"threshold": threshold}, name=name))
        return self

    def add_linear(self, weight, bias=None, name=""):
        self.add(SNNLayer("linear", {
            "weight": weight.clone(),
            "bias": bias.clone() if bias is not None else None,
        }, name=name))
        return self

    def add_avgpool(self, output_size=1, name=""):
        self.add(SNNLayer("avgpool", {"output_size": output_size}, name=name))
        return self

    def add_flatten(self, name=""):
        self.add(SNNLayer("flatten", {}, name=name))
        return self

    def to_dict(self) -> dict:
        """Export to JSON-serializable dict (NIR-compatible)."""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "layers": [
                {
                    "type": l.layer_type,
                    "name": l.name,
                    "params": {k: v.tolist() if isinstance(v, torch.Tensor) else v
                               for k, v in l.params.items()},
                    "input_shape": l.input_shape,
                    "output_shape": l.output_shape,
                }
                for l in self.layers
            ],
        }

    def save(self, path: str):
        """Save IR to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_snn_model(cls, snn_model, T=64) -> "SNNGraph":
        """Extract IR from a converted SNN model."""
        graph = cls(name="converted_snn")
        graph.metadata["T"] = T

        for name, module in snn_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                graph.add_conv2d(
                    module.weight,
                    module.bias,
                    stride=module.stride[0],
                    padding=module.padding[0],
                    name=name,
                )
            elif isinstance(module, torch.nn.Linear):
                graph.add_linear(module.weight, module.bias, name=name)
            elif isinstance(module, torch.nn.AvgPool2d):
                graph.add_avgpool(name=name)
            elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
                graph.add_avgpool(output_size=module.output_size[0], name=name)
            elif isinstance(module, torch.nn.Flatten):
                graph.add_flatten(name=name)

        # Add IF neurons (extracted from snnTorch Leaky modules)
        import snntorch as snn
        for name, module in snn_model.named_modules():
            if isinstance(module, snn.Leaky):
                graph.add_if_neuron(
                    threshold=module.threshold if hasattr(module, 'threshold') else 1.0,
                    name=name,
                )

        return graph
