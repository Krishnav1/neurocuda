"""
Optional Intel Lava SDK integration helpers.

Lava (lava-nc) requires Python <3.11 and INRC access for Loihi2HwCfg.
When unavailable, NeuroCUDA uses Loihi2Sim-equivalent validation via the
internal loihi quant backend (clearly labeled in run metadata).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

_LAVA_AVAILABLE: Optional[bool] = None
_NIR_TO_LAVA_AVAILABLE: Optional[bool] = None


def lava_available() -> bool:
    global _LAVA_AVAILABLE
    if _LAVA_AVAILABLE is None:
        try:
            import lava  # noqa: F401

            _LAVA_AVAILABLE = True
        except ImportError:
            _LAVA_AVAILABLE = False
    return _LAVA_AVAILABLE


def nir_to_lava_available() -> bool:
    global _NIR_TO_LAVA_AVAILABLE
    if _NIR_TO_LAVA_AVAILABLE is None:
        try:
            from nir_to_lava import import_from_nir  # noqa: F401

            _NIR_TO_LAVA_AVAILABLE = True
        except ImportError:
            _NIR_TO_LAVA_AVAILABLE = False
    return _NIR_TO_LAVA_AVAILABLE


def import_nir_to_lava(
    nir_graph: Any,
    *,
    dt: float = 1e-4,
    fixed_pt: bool = True,
    on_chip: bool = False,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    Import official NIR graph into Lava processes.

    Returns (lava_nodes, start_nodes, end_nodes, config_dict).
    """
    if not lava_available() or not nir_to_lava_available():
        raise ImportError(
            "Lava SDK + nir_to_lava required (Python 3.10, INRC Linux). "
            "See docs/LAVA_SETUP.md"
        )

    from nir_to_lava import ImportConfig, LavaLibrary, import_from_nir

    config = ImportConfig(
        dt=dt,
        fixed_pt=fixed_pt,
        on_chip=on_chip,
        library_preference=LavaLibrary.Lava,
    )
    lava_nodes, start_nodes, end_nodes = import_from_nir(nir_graph, config)
    return lava_nodes, start_nodes, end_nodes, {
        "dt": dt,
        "fixed_pt": fixed_pt,
        "on_chip": on_chip,
    }


def compare_spike_traces(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Compare two binary spike traces."""
    a = np.asarray(a).astype(int).flatten()
    b = np.asarray(b).astype(int).flatten()
    n = min(len(a), len(b))
    if n == 0:
        return {"compared": 0, "diffs": 0, "match_rate": 1.0}
    diffs = int(np.sum(a[:n] != b[:n]))
    return {
        "compared": n,
        "diffs": diffs,
        "match_rate": float(1.0 - diffs / n),
    }
