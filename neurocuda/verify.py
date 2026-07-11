"""
NeuroCUDA Cross-Backend Verification
======================================
nc.verify() — accuracy and deployment parity across GPU/CPU/Loihi/Lava paths.

GATE L2 targets (MLP MNIST):
  - Reference backend accuracy gap <= 2%
  - Minimum accuracy >= 95.4% (when using hub-quality weights)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .backends import get_backend

try:
    from models import IFNeuron, LIFNeuron, reset_spiking
except ImportError:
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from models import IFNeuron, LIFNeuron, reset_spiking


DEFAULT_BACKENDS = ("gpu", "cpu", "loihi", "loihi2_lava")

GATE_L2_MIN_ACCURACY = 95.4
GATE_L2_MAX_GAP_PCT = 2.0


def _accuracy_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    T: int = 32,
    backend_name: str = "gpu",
) -> Dict[str, float]:
    """Evaluate classification accuracy with temporal spiking if needed."""
    model = model.to(device).eval()
    correct, total = 0, 0
    is_spiking = any(isinstance(m, (IFNeuron, LIFNeuron)) for m in model.modules())

    use_backend = backend_name in ("loihi", "loihi2_lava", "loihi2_hw")
    backend = None
    compiled = None
    if use_backend:
        backend = get_backend(backend_name)
        compiled = backend.compile(model, T=T)

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)

            if compiled is not None:
                out = backend.run(compiled, data, T=T)
            elif is_spiking:
                reset_spiking(model)
                if data.dim() == 2:
                    acc = torch.zeros(data.size(0), _out_features(model), device=device)
                    for _ in range(T):
                        acc += model(data)
                    out = acc / T
                elif data.dim() == 4:
                    acc = torch.zeros(data.size(0), _out_features(model), device=device)
                    for _ in range(T):
                        acc += model(data)
                    out = acc / T
                else:
                    out = model(data)
            else:
                out = model(data)

            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc_pct = 100.0 * correct / max(total, 1)
    return {"accuracy": acc_pct, "correct": correct, "total": total}


def _out_features(model: nn.Module) -> int:
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Linear):
            return m.out_features
    return 10


def verify(
    snn_model: nn.Module,
    test_loader: DataLoader,
    backends: Optional[Sequence[str]] = None,
    reference: str = "gpu",
    T: int = 32,
    device: Optional[torch.device] = None,
    gate_l2: bool = True,
    min_accuracy: float = GATE_L2_MIN_ACCURACY,
    max_gap_pct: float = GATE_L2_MAX_GAP_PCT,
) -> Dict[str, Any]:
    """
    Compare SNN accuracy across multiple NeuroCUDA backends.

    Returns a report dict with per-backend metrics and gate pass/fail.
    """
    if backends is None:
        backends = list(DEFAULT_BACKENDS)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "T": T,
        "reference": reference,
        "backends": {},
        "gates": {},
        "passed": False,
    }

    ref_acc = None
    for name in backends:
        try:
            metrics = _accuracy_on_loader(
                snn_model, test_loader, device, T=T, backend_name=name
            )
            entry: Dict[str, Any] = {"status": "ok", **metrics}
            if name == "loihi2_lava":
                try:
                    be = get_backend(name)
                    compiled = be.compile(snn_model, T=T)
                    entry["execution_mode"] = getattr(
                        compiled, "execution_mode", "unknown"
                    )
                except Exception as exc:
                    entry["execution_mode"] = f"compile_error: {exc}"
            report["backends"][name] = entry
            if name == reference:
                ref_acc = metrics["accuracy"]
        except Exception as exc:
            report["backends"][name] = {"status": "error", "error": str(exc)}

    if ref_acc is None and reference in report["backends"]:
        ref_acc = report["backends"][reference].get("accuracy")

    gaps: Dict[str, float] = {}
    if ref_acc is not None:
        for name, entry in report["backends"].items():
            if entry.get("status") == "ok" and "accuracy" in entry:
                gaps[name] = ref_acc - entry["accuracy"]
        report["gaps_vs_reference"] = gaps

    # GATE L2 checks
    l2_pass = True
    l2_reasons: List[str] = []
    if gate_l2 and ref_acc is not None:
        for name, entry in report["backends"].items():
            if entry.get("status") != "ok":
                if name in ("loihi2_lava", "loihi2_hw"):
                    l2_pass = False
                    l2_reasons.append(f"{name}: {entry.get('error', entry.get('status'))}")
                continue
            acc = entry["accuracy"]
            gap = abs(gaps.get(name, 0.0))
            if name in ("loihi2_lava", "loihi2_hw", "loihi"):
                if acc < min_accuracy:
                    l2_pass = False
                    l2_reasons.append(f"{name} accuracy {acc:.2f}% < {min_accuracy}%")
                if gap > max_gap_pct:
                    l2_pass = False
                    l2_reasons.append(
                        f"{name} gap {gap:.2f}% > {max_gap_pct}% vs {reference}"
                    )

    report["gates"]["L2"] = {
        "passed": l2_pass,
        "min_accuracy": min_accuracy,
        "max_gap_pct": max_gap_pct,
        "reasons": l2_reasons,
    }
    report["passed"] = l2_pass and all(
        report["backends"].get(b, {}).get("status") == "ok"
        for b in ("gpu", "cpu")
    )

    return report


def verify_to_json(report: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Write verify report to JSON (NeuroBench / arXiv supplement)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path
