#!/usr/bin/env python3
"""
Run MLP MNIST on physical Loihi 2 via Lava Loihi2HwCfg (INRC required).

Prerequisites:
  - INRC membership + Loihi Magma extension installed
  - Python 3.10 Linux environment
  - export INRC_LOIHI=1

Usage:
    python scripts/run_loihi_hw_benchmark.py --hub mlp-mnist-snn
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import neurocuda as nc
from neurocuda.backends.loihi2_lava import Loihi2LavaBackend
from neurocuda.verify import verify, verify_to_json, GATE_L2_MIN_ACCURACY


def main():
    if os.environ.get("INRC_LOIHI", "") != "1":
        print("Set INRC_LOIHI=1 and run on INRC Loihi cloud / Linux Py3.10 with Lava SDK.")
        print("Dry-run: use reproduce.py --lava-gate instead.")
        sys.exit(2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", default="mlp-mnist-snn")
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--out", default="results/loihi_hw_benchmark.json")
    args = parser.parse_args()

    if not Loihi2LavaBackend.sdk_available():
        print("Lava SDK + nir_to_lava not available.")
        sys.exit(1)

    model, info = nc.hub.load(f"neurocuda/{args.hub}" if "/" not in args.hub else args.hub)
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_ds = datasets.MNIST(str(ROOT / "data"), train=False, download=True, transform=tf)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    backend = Loihi2LavaBackend(fixed_pt=True, on_chip=True)
    compiled = backend.compile(model, T=args.T)
    print(f"Execution mode: {compiled.execution_mode}")

    report = verify(
        model,
        loader,
        backends=["gpu", "loihi2_hw"],
        reference="gpu",
        T=args.T,
        min_accuracy=GATE_L2_MIN_ACCURACY,
    )
    report["hub_info"] = {k: info.get(k) for k in ("name", "if_accuracy") if isinstance(info, dict)}
    report["hardware"] = "loihi2_physical_inrc"

    out = verify_to_json(report, ROOT / args.out)
    print(f"GATE L4 report: {out}")
    print(f"L2 passed: {report['gates']['L2']['passed']}")
    for name, entry in report["backends"].items():
        if entry.get("status") == "ok":
            print(f"  {name}: {entry['accuracy']:.2f}%")
    sys.exit(0 if report["gates"]["L2"]["passed"] else 1)


if __name__ == "__main__":
    main()
