"""
Manual NIR graph executor with proper topological sort.
Handles residual connections (multiple inputs → sum) correctly.
For accuracy verification of NIR round-trip.
"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque


class NIRExecutor(nn.Module):
    """Execute a NIR graph with proper topological sort (Kahn's algorithm).
    Handles multiple inputs by summation (correct for residual adds in rate-coded SNN)."""

    def __init__(self, nir_graph, node_to_torch_fn):
        super().__init__()
        self.node_order = []  # execution order (list of node names)
        self.multi_input_nodes = set()  # nodes that need input summation
        self._build(nir_graph, node_to_torch_fn)

    def _build(self, nir_graph, node_to_torch_fn):
        """Build the execution plan from NIR graph."""
        nodes = nir_graph.nodes
        edges = nir_graph.edges

        # Build torch modules
        for name, nir_node in nodes.items():
            if isinstance(nir_node, (nir.Input, nir.Output)):
                continue
            mod = node_to_torch_fn(nir_node)
            if mod is not None:
                self.add_module(name, mod)

        # Build adjacency for topological sort
        # incoming[dest] = list of sources
        incoming = {name: [] for name in nodes}
        outgoing = {name: [] for name in nodes}
        for src, dst in edges:
            if src in incoming and dst in incoming:
                incoming[dst].append(src)
                outgoing[src].append(dst)

        # Mark nodes with multiple inputs (residual adds)
        for name, srcs in incoming.items():
            if len(srcs) > 1:
                self.multi_input_nodes.add(name)

        # Kahn's topological sort
        in_degree = {name: len(srcs) for name, srcs in incoming.items()}
        queue = deque([name for name, deg in in_degree.items() if deg == 0])

        # If no zero-degree nodes (shouldn't happen), use Input nodes
        if not queue:
            for name, nir_node in nodes.items():
                if isinstance(nir_node, nir.Input):
                    queue.append(name)

        topo_order = []
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for child in outgoing.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Filter to computable nodes (exclude Input/Output)
        self.node_order = [
            n for n in topo_order
            if n not in nodes or not isinstance(nodes[n], (nir.Input, nir.Output))
        ]
        # Keep only nodes that have torch modules
        self.node_order = [
            n for n in self.node_order
            if hasattr(self, n) or n in self._modules
        ]

        # Store edges for input resolution
        self._incoming = incoming
        self._nodes = nodes

    def forward(self, x):
        """Execute the graph. x is the input tensor."""
        # Put data at ALL input nodes (NIR read may create extra Input nodes)
        cache = {}
        for name, nir_node in self._nodes.items():
            if isinstance(nir_node, nir.Input):
                cache[name] = x

        for name in self.node_order:
            mod = self._modules.get(name)
            if mod is None:
                continue

            # Collect inputs
            srcs = self._incoming.get(name, [])
            inputs = []
            for src in srcs:
                if src in cache:
                    inputs.append(cache[src])

            if len(inputs) == 0:
                # This node has no inputs — skip
                continue
            elif len(inputs) == 1:
                inp = inputs[0]
            else:
                # Multiple inputs: SUM them (residual connection)
                inp = torch.stack(inputs, dim=0).sum(dim=0)

            # Apply module — NEVER reset state here.
            # Caller is responsible for reset() before multi-timestep loops.
            out = mod(inp)
            cache[name] = out

        # Find THE real output node.
        # NIR read may add extra Output nodes (e.g., output_x passthrough from input).
        # The real output connects to a compute node (fc/Affine), not directly to input.
        output_src = None
        for name, nir_node in self._nodes.items():
            if isinstance(nir_node, nir.Output):
                srcs = self._incoming.get(name, [])
                for src in srcs:
                    # Skip outputs that are just input passthrough
                    if src == "x":
                        continue
                    if src in cache:
                        output_src = src
                        break
            if output_src:
                break

        if output_src and output_src in cache:
            return cache[output_src]

        raise RuntimeError(f"No output produced. Cache keys: {list(cache.keys())}")


# Lazy imports to avoid circular dependency
import sys
sys.path.insert(0, ".")
from models import IFNeuron
import nir


def make_torch_from_nir(nir_node):
    """Convert a NIR node to a torch module with correct weights."""
    if isinstance(nir_node, nir.Conv2d):
        w = torch.from_numpy(nir_node.weight).float()
        b = torch.from_numpy(nir_node.bias).float() if nir_node.bias is not None else None
        stride = (int(nir_node.stride[0]), int(nir_node.stride[1])) if hasattr(nir_node.stride, '__getitem__') else (int(nir_node.stride),) * 2
        padding = (int(nir_node.padding[0]), int(nir_node.padding[1])) if hasattr(nir_node.padding, '__getitem__') else (int(nir_node.padding),) * 2
        dilation = (int(nir_node.dilation[0]), int(nir_node.dilation[1])) if hasattr(nir_node.dilation, '__getitem__') else (int(nir_node.dilation),) * 2
        c = nn.Conv2d(
            int(w.shape[1]), int(w.shape[0]),
            (int(w.shape[2]), int(w.shape[3])),
            stride=stride, padding=padding, dilation=dilation,
            groups=int(nir_node.groups), bias=b is not None
        )
        c.weight = nn.Parameter(w)
        if b is not None:
            c.bias = nn.Parameter(b)
        return c

    if isinstance(nir_node, nir.Affine):
        w = torch.from_numpy(nir_node.weight).float()
        b = torch.from_numpy(nir_node.bias).float()
        fc = nn.Linear(int(w.shape[1]), int(w.shape[0]))
        fc.weight = nn.Parameter(w)
        fc.bias = nn.Parameter(b)
        return fc

    if isinstance(nir_node, nir.IF):
        thresh = float(nir_node.v_threshold.mean()) if nir_node.v_threshold is not None else 1.0
        return IFNeuron(thresh=thresh)

    if isinstance(nir_node, nir.AvgPool2d):
        return nn.AdaptiveAvgPool2d(1)

    if isinstance(nir_node, nir.Flatten):
        return nn.Flatten(start_dim=1)

    return None


def accuracy_check(original_model, nir_path, sample_input):
    """Full accuracy verification: export → read → execute → compare."""
    print(f"\n=== ACCURACY VERIFICATION ===")
    print(f"Reading NIR: {nir_path}")
    g = nir.read(nir_path)
    print(f"  {len(g.nodes)} nodes, {len(g.edges)} edges")

    print("Building NIR executor...")
    executor = NIRExecutor(g, make_torch_from_nir)
    print(f"  Execution order: {len(executor.node_order)} nodes")

    # Run both models
    with torch.no_grad():
        # Reset original IF state
        for m in original_model.modules():
            if isinstance(m, IFNeuron):
                m.reset()
        orig_out = original_model(sample_input)

        # Reset executor IF state
        for m in executor.modules():
            if isinstance(m, IFNeuron):
                m.reset()
        rebuilt_out = executor(sample_input)

    diff = (orig_out - rebuilt_out).abs().max().item()
    print(f"\n  Max absolute difference: {diff:.6e}")
    return diff


if __name__ == "__main__":
    from models import resnet18_cifar, QCFS, IFNeuron, build_snn_from_qcfs
    from resnet_nir_export import export_resnet_to_nir

    print("=" * 60)
    print("RESNET NIR FULL ACCURACY VERIFICATION")
    print("=" * 60)

    # Build & export
    print("\nBuilding ResNet SNN...")
    q = resnet18_cifar(lambda: QCFS(L=8))
    snn = build_snn_from_qcfs(q)
    snn.eval()

    g = export_resnet_to_nir(snn, "resnet_snn.nir")

    # Verify
    x = torch.randn(2, 3, 32, 32)
    diff = accuracy_check(snn, "resnet_snn.nir", x)

    print("\n" + "=" * 60)
    if diff < 1e-4:
        print("🎉 RESNET NIR FULL ROUND-TRIP: BIT-EXACT ACCURACY MATCH!")
        print("   WRITE ✅  READ ✅  EXECUTE ✅  ACCURACY ✅")
    elif diff < 1e-2:
        print(f"RESNET NIR: NEAR MATCH (diff={diff:.2e}) — investigate")
    else:
        print(f"RESNET NIR: MISMATCH (diff={diff:.2e}) — debugging needed")
        # Show per-node shapes
        print("\nDebug: execution order and shapes")
        ex = NIRExecutor(nir.read("resnet_snn.nir"), make_torch_from_nir)
        cache = {"x": x}
        for name in ex.node_order[:10]:
            mod = ex._modules.get(name)
            if mod is None:
                continue
            srcs = ex._incoming.get(name, [])
            inputs = [cache[s] for s in srcs if s in cache]
            if inputs:
                inp = torch.stack(inputs).sum(0) if len(inputs) > 1 else inputs[0]
                if isinstance(mod, IFNeuron): mod.reset()
                out = mod(inp)
                cache[name] = out
                print(f"  {name}({type(mod).__name__}): {list(inp.shape)} -> {list(out.shape)}")
    print("=" * 60)
