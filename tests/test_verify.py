"""Tests for nc.verify() cross-backend verification."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import IFNeuron, reset_spiking
import neurocuda as nc
from neurocuda.verify import verify, GATE_L2_MAX_GAP_PCT


class TinyMLP(nn.Module):
    def __init__(self, n_in=16, n_h=8, n_out=3):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_h)
        self.if1 = IFNeuron(thresh=0.5)
        self.fc2 = nn.Linear(n_h, n_out)

    def forward(self, x):
        return self.fc2(self.if1(self.fc1(x)))


@pytest.fixture
def tiny_loader():
    torch.manual_seed(0)
    x = torch.randn(32, 16)
    y = torch.randint(0, 3, (32,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_verify_runs_all_backends(tiny_loader):
    model = TinyMLP()
    report = verify(
        model,
        tiny_loader,
        backends=["gpu", "cpu", "loihi"],
        T=4,
        gate_l2=False,
    )
    assert report["backends"]["gpu"]["status"] == "ok"
    assert report["backends"]["cpu"]["status"] == "ok"
    assert "accuracy" in report["backends"]["gpu"]


def test_verify_loihi2_lava_backend(tiny_loader):
    pytest.importorskip("nir")
    model = TinyMLP()
    report = verify(
        model,
        tiny_loader,
        backends=["gpu", "loihi2_lava"],
        T=4,
        gate_l2=False,
    )
    assert report["backends"]["loihi2_lava"]["status"] == "ok"
    gap = abs(report["gaps_vs_reference"].get("loihi2_lava", 0))
    assert gap <= 10.0  # tiny random model — loose bound


def test_verify_gate_l2_structure(tiny_loader):
    model = TinyMLP()
    report = verify(model, tiny_loader, T=4, gate_l2=True, min_accuracy=0.0, max_gap_pct=100.0)
    assert "L2" in report["gates"]
    assert "passed" in report["gates"]["L2"]
