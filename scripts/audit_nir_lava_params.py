#!/usr/bin/env python3
"""
Audit NIR neuron parameters before/after Lava import.

Detects threshold/decay shifts (NIR issue #111) that cause accuracy drops on fixed_pt.

Usage:
    python scripts/audit_nir_lava_params.py --model-path checkpoints/hub/mlp_mnist_snn.pt
    python scripts/audit_nir_lava_params.py --hub mlp-mnist-snn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from neurocuda.backends.nir_bridge import audit_nir_params, snn_to_nir_graph
from neurocuda.backends.lava_utils import lava_available, nir_to_lava_available


def build_mlp_from_state(state_dict: dict) -> nn.Module:
    """Reconstruct MLP MNIST SNN from hub checkpoint keys."""
    from models import IFNeuron

    class MLPMNIST(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(784, 256)
            self.if1 = IFNeuron(thresh=1.0)
            self.fc2 = nn.Linear(256, 256)
            self.if2 = IFNeuron(thresh=1.0)
            self.fc3 = nn.Linear(256, 10)

        def forward(self, x):
            x = self.flatten(x)
            x = self.if1(self.fc1(x))
            x = self.if2(self.fc2(x))
            return self.fc3(x)

    model = MLPMNIST()
    model.load_state_dict(state_dict, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="Audit NIR params for Lava deployment")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--hub", type=str, default=None, help="Hub model name e.g. mlp-mnist-snn")
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--out", type=str, default="results/nir_param_audit.json")
    args = parser.parse_args()

    if args.hub:
        import neurocuda as nc

        model, _info = nc.hub.load(args.hub)
    elif args.model_path:
        state = torch.load(args.model_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model = build_mlp_from_state(state)
    else:
        print("Provide --model-path or --hub")
        sys.exit(1)

    model.eval()
    graph = snn_to_nir_graph(model, T=args.T)
    audit = audit_nir_params(graph)

    report = {
        "lava_sdk": lava_available(),
        "nir_to_lava": nir_to_lava_available(),
        "T": args.T,
        "nir_audit": audit,
    }

    if nir_to_lava_available() and lava_available():
        from neurocuda.backends.lava_utils import import_nir_to_lava

        _, _, _, cfg = import_nir_to_lava(graph, fixed_pt=True, on_chip=False)
        report["lava_import_config"] = cfg
        print("Lava import: OK (fixed_pt=True, on_chip=False)")
    else:
        print("Lava SDK not available — NIR audit only (install on INRC Py3.10 Linux)")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {out_path}")
    for node in audit["nodes"]:
        if "v_threshold" in node:
            vt = node["v_threshold"]
            print(
                f"  {node['name']:12} {node['type']:6} "
                f"v_threshold mean={vt['mean']:.4f} shape={vt['shape']}"
            )


if __name__ == "__main__":
    main()
