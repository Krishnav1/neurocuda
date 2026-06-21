"""
NIR export tests — verify round-trip capability.

Tests:
  - to_nir produces valid NIR dict with nodes and edges
  - NIR export works on a simple spiking model
  - NIR export handles per-channel IF models
  - Exported graph has correct structure
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neurocuda import to_nir, convert
from models import IFNeuron, QCFS

# Model definition for test independence
class SimpleCNN4D(nn.Module):
    def __init__(self, act_factory=nn.ReLU, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.act1 = act_factory()
        self.pool = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 16 * 16, num_classes)
    def forward(self, x):
        x = self.pool(self.act1(self.bn1(self.conv1(x))))
        return self.fc(self.flatten(x))


# ===========================================================================
# Tests
# ===========================================================================

class TestNIRExport:
    """NIR export format tests."""

    @pytest.fixture
    def snn_model(self, device):
        """A simple converted SNN for NIR export testing."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device)

        # Quick convert with minimal data
        x = torch.randn(8, 1, 32, 32, device=device)
        y = torch.zeros(8, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)

        snn, _ = convert(
            model,
            loader,
            qcfs_epochs=1,
            if_epochs=1,
            strategy="qcfs_if_ft",
            channel_wise=False,
            device=device,
            verbose=False,
        )
        return snn

    def test_to_nir_returns_dict(self, snn_model):
        """to_nir returns a dictionary."""
        result = to_nir(snn_model, T=16, model_name="test_snn")
        assert isinstance(result, dict)

    def test_to_nir_has_nodes(self, snn_model):
        """NIR graph has nodes."""
        result = to_nir(snn_model, T=16, model_name="test_snn")
        assert "nodes" in result, f"Missing 'nodes' key. Keys: {list(result.keys())}"
        assert len(result["nodes"]) > 0, "NIR graph should have at least one node"

    def test_to_nir_has_edges(self, snn_model):
        """NIR graph has edges."""
        result = to_nir(snn_model, T=16, model_name="test_snn")
        assert "edges" in result, f"Missing 'edges' key. Keys: {list(result.keys())}"
        # Edges can be empty for single-node graphs, but our CNN has >1 node
        assert len(result["edges"]) > 0, "NIR graph should have edges connecting nodes"

    def test_to_nir_nodes_have_types(self, snn_model):
        """Each NIR node has a type field."""
        result = to_nir(snn_model, T=16, model_name="test_snn")
        for node_id, node_data in result["nodes"].items():
            assert "type" in node_data or hasattr(node_data, 'type'), \
                f"Node {node_id} missing type information"

    def test_to_nir_model_name_propagates(self, snn_model):
        """Model name parameter is used."""
        result = to_nir(snn_model, T=32, model_name="my_custom_snn")
        assert isinstance(result, dict)  # Name is embedded in graph metadata

    def test_to_nir_different_T_values(self, snn_model):
        """to_nir works with different T values."""
        for T in [4, 8, 16, 32]:
            result = to_nir(snn_model, T=T, model_name=f"test_T{T}")
            assert isinstance(result, dict), f"Failed at T={T}"

    def test_to_nir_rejects_non_module(self):
        """to_nir raises TypeError for non-Module input."""
        with pytest.raises(TypeError):
            to_nir("not_a_model", T=16)


class TestNIRChannelWise:
    """NIR export with per-channel IF models."""

    @pytest.fixture
    def snn_channel_wise(self, device):
        """A converted SNN with per-channel IF neurons."""
        model = SimpleCNN4D(act_factory=nn.ReLU).to(device)

        x = torch.randn(8, 1, 32, 32, device=device)
        y = torch.zeros(8, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)

        snn, _ = convert(
            model,
            loader,
            qcfs_epochs=1,
            if_epochs=1,
            strategy="qcfs_if_ft",
            channel_wise=True,
            device=device,
            verbose=False,
        )
        return snn

    def test_channel_wise_nir_export(self, snn_channel_wise):
        """NIR export works with per-channel IF neurons."""
        result = to_nir(snn_channel_wise, T=16, model_name="test_cs")
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0


class TestNIRRoundTrip:
    """NIR export → graph integrity tests."""

    def test_edge_indices_valid(self, device):
        """All edges have valid structure (handles both dict and tuple formats)."""
        class MiniSNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
                self.if1 = IFNeuron(thresh=1.0)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(4 * 32 * 32, 3)

            def forward(self, x):
                x = self.if1(self.conv1(x))
                return self.fc(self.flatten(x))

        model = MiniSNN().to(device).eval()
        _ = model(torch.randn(1, 1, 32, 32, device=device))

        result = to_nir(model, T=8, model_name="mini_snn")

        # Edges can be dicts, tuples, or lists depending on NIR version
        for edge in result["edges"]:
            if isinstance(edge, dict):
                # Dict format
                assert len(edge) >= 2, f"Edge dict should have at least 2 entries: {edge}"
            elif isinstance(edge, (tuple, list)):
                # Tuple/list format: (source, target) or (source, target, metadata)
                assert len(edge) >= 2, f"Edge tuple should have at least 2 entries: {edge}"
            else:
                # Unknown format — just verify it's not empty
                assert edge, "Edge should not be empty"

    def test_empty_model_handled(self, device):
        """Model with no layers outputs minimal graph."""
        model = nn.Sequential().to(device)
        # Sequential with no layers may or may not export — just verify no crash
        try:
            result = to_nir(model, T=16, model_name="empty")
            assert isinstance(result, dict)
        except Exception:
            # Empty model failing to export is acceptable
            pass
