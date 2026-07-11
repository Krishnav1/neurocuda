"""
Lava integration tests — GATE L0/L1/L2 validation.

Runs without full Lava SDK when unavailable (uses NeuroCUDA Loihi sim fallback).
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import IFNeuron, reset_spiking
from neurocuda.backends.nir_bridge import audit_nir_params, nir_available, snn_to_nir_graph
from neurocuda.backends.loihi2_lava import Loihi2LavaBackend
from neurocuda.backends.lava_utils import lava_available
from tests.test_lava_equivalence import validate_equivalence


pytestmark = pytest.mark.skipif(not nir_available(), reason="nir package not installed")


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.if1 = IFNeuron(thresh=1.0)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.if1(self.fc1(x))
        return self.fc2(x)


def test_gate_l0_nir_graph_builds():
    model = TinyMLP()
    graph = snn_to_nir_graph(model, T=16, type_check=True)
    audit = audit_nir_params(graph)
    assert len(audit["nodes"]) >= 3
    assert graph.input_type is not None


def test_gate_l1_single_if_neuron():
    diffs, total = validate_equivalence(threshold=1.0, n_neurons=500, n_steps=50)
    assert diffs == 0, f"Expected 0 spike diffs, got {diffs}/{total}"


def test_loihi2_lava_compile_and_run():
    model = TinyMLP()
    backend = Loihi2LavaBackend(fixed_pt=True, on_chip=False)
    compiled = backend.compile(model, T=8)
    assert compiled.nir_graph is not None
    x = torch.randn(4, 8)
    out = backend.run(compiled, x, T=8)
    assert out.shape == (4, 2)


def test_loihi2_lava_backend_registered():
    import neurocuda as nc

    assert "loihi2_lava" in nc.list_backends()


def test_lava_sdk_status_reported():
    backend = Loihi2LavaBackend()
    compiled = backend.compile(TinyMLP(), T=4)
    if not Loihi2LavaBackend.sdk_available():
        assert compiled.execution_mode == "neurocuda_loihi_sim"
    else:
        assert compiled.execution_mode.startswith("lava_")
