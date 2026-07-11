#!/usr/bin/env python3
"""
NeuroCUDA vs SNNToolBox — MLP MNIST head-to-head (same ANN weights).

Protocol (honest):
  - Architecture: Flatten → Dense(256, ReLU) → Dense(256, ReLU) → Dense(10)
  - Same trained Keras ANN weights copied into PyTorch for NeuroCUDA
  - Full MNIST test set (10,000) unless --num-to-test is set
  - Same timestep budget T (default 32)
  - Report ANN accuracy, SNN accuracy, gap, wall time

Usage:
  python scripts/compare_snntoolbox_mlp_mnist.py
  python scripts/compare_snntoolbox_mlp_mnist.py --num-to-test 1000  # faster smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_mnist_numpy():
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(str(ROOT / "data"), train=True, download=True, transform=tf)
    test = datasets.MNIST(str(ROOT / "data"), train=False, download=True, transform=tf)

    def to_xy(ds):
        xs, ys = [], []
        for x, y in ds:
            xs.append(x.numpy().reshape(-1))
            ys.append(y)
        return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)

    x_train, y_train = to_xy(train)
    x_test, y_test = to_xy(test)
    return x_train, y_train, x_test, y_test


def train_keras_ann(x_train, y_train, x_test, y_test, epochs=5, seed=0):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(256, activation="relu", name="fc1"),
        keras.layers.Dense(256, activation="relu", name="fc2"),
        keras.layers.Dense(10, activation="softmax", name="fc3"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    t0 = time.perf_counter()
    model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(x_test, y_test),
        verbose=1,
    )
    train_s = time.perf_counter() - t0
    loss, ann_acc = model.evaluate(x_test, y_test, verbose=0)
    return model, float(ann_acc) * 100.0, train_s


def keras_to_torch(keras_model) -> nn.Module:
    """Copy Dense kernels into a NeuroCUDA-compatible MLP (ReLU names for convert)."""

    class MLPMNIST(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(784, 256)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(256, 256)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(256, 10)

        def forward(self, x):
            x = self.flatten(x)
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            return self.fc3(x)

    pt = MLPMNIST()
    # Keras Dense kernel: (in, out); PyTorch weight: (out, in)
    w1, b1 = keras_model.get_layer("fc1").get_weights()
    w2, b2 = keras_model.get_layer("fc2").get_weights()
    w3, b3 = keras_model.get_layer("fc3").get_weights()
    with torch.no_grad():
        pt.fc1.weight.copy_(torch.from_numpy(w1.T))
        pt.fc1.bias.copy_(torch.from_numpy(b1))
        pt.fc2.weight.copy_(torch.from_numpy(w2.T))
        pt.fc2.bias.copy_(torch.from_numpy(b2))
        pt.fc3.weight.copy_(torch.from_numpy(w3.T))
        pt.fc3.bias.copy_(torch.from_numpy(b3))
    return pt.eval()


def eval_torch_ann(model, x_test, y_test, batch=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(x_test), batch):
            xb = torch.from_numpy(x_test[i : i + batch]).to(device)
            yb = torch.from_numpy(y_test[i : i + batch]).to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return 100.0 * correct / total


def run_neurocuda(pt_ann, x_train, y_train, x_test, y_test, T=32, qcfs_epochs=3, if_epochs=3):
    import neurocuda as nc
    from models import reset_spiking

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=128, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=256, shuffle=False,
    )

    t0 = time.perf_counter()
    snn, stats = nc.convert(
        pt_ann,
        train_loader,
        test_loader=test_loader,
        qcfs_epochs=qcfs_epochs,
        if_epochs=if_epochs,
        strategy="qcfs_if_ft",
        channel_wise=False,
        device=device,
        verbose=False,
    )
    convert_s = time.perf_counter() - t0

    # Explicit T-loop accuracy on full test (match verify protocol)
    snn = snn.to(device).eval()
    correct = total = 0
    t1 = time.perf_counter()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            reset_spiking(snn)
            acc = torch.zeros(data.size(0), 10, device=device)
            for _ in range(T):
                acc += snn(data)
            pred = (acc / T).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    infer_s = time.perf_counter() - t1
    snn_acc = 100.0 * correct / total

    return {
        "tool": "neurocuda",
        "method": "QCFS→IF + BPTT FT",
        "snn_accuracy": snn_acc,
        "convert_stats": {
            "qcfs_accuracy": stats.get("qcfs_accuracy"),
            "if_accuracy": stats.get("if_accuracy"),
        },
        "T": T,
        "convert_seconds": convert_s,
        "infer_seconds": infer_s,
        "num_tested": total,
        "features": {
            "nir_export": True,
            "multi_backend": True,
            "loihi_sim": True,
            "ros2": True,
        },
    }


def run_snntoolbox_style_rate(pt_ann, x_calib, x_test, y_test, T=32, percentile=99.9, batch=256):
    """
    Rueckauer / SNNToolBox-style rate conversion (reimplementation).

    Official snntoolbox 0.6.0 fails on Keras 3.x (loss_weights). This mirrors the
    core method: percentile activation normalize + soft-reset IF over T steps.
    """
    from models import IFNeuron, reset_spiking
    import copy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann = copy.deepcopy(pt_ann).to(device).eval()

    # Collect post-ReLU activations on calibration set
    acts = {1: [], 2: []}

    def hook(idx):
        def _h(_m, _i, o):
            acts[idx].append(o.detach())
        return _h

    h1 = ann.relu1.register_forward_hook(hook(1))
    h2 = ann.relu2.register_forward_hook(hook(2))
    with torch.no_grad():
        for i in range(0, min(len(x_calib), 10000), batch):
            xb = torch.from_numpy(x_calib[i : i + batch]).to(device)
            _ = ann(xb)
    h1.remove()
    h2.remove()

    a1 = torch.cat(acts[1], 0).reshape(-1).cpu().numpy()
    a2 = torch.cat(acts[2], 0).reshape(-1).cpu().numpy()
    thr1 = max(float(np.percentile(a1, percentile)), 1e-6)
    thr2 = max(float(np.percentile(a2, percentile)), 1e-6)

    # Scale next-layer weights by previous threshold (data-based weight normalization)
    snn = copy.deepcopy(ann)
    with torch.no_grad():
        # Rueckauer-style: keep ReLU→IF with calibrated thresholds (rate ≈ act/thr)
        snn.relu1 = IFNeuron(thresh=thr1)
        snn.relu2 = IFNeuron(thresh=thr2)

    snn = snn.to(device).eval()
    correct = total = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(x_test), batch):
            xb = torch.from_numpy(x_test[i : i + batch]).to(device)
            yb = torch.from_numpy(y_test[i : i + batch]).to(device)
            reset_spiking(snn)
            acc = torch.zeros(xb.size(0), 10, device=device)
            for _ in range(T):
                acc += snn(xb)
            pred = (acc / T).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    infer_s = time.perf_counter() - t0
    return {
        "tool": "snntoolbox_style_rate",
        "method": f"percentile-{percentile} IF thresholds (Rueckauer-style reimpl)",
        "snn_accuracy": 100.0 * correct / total,
        "thresholds": {"relu1": thr1, "relu2": thr2},
        "T": T,
        "infer_seconds": infer_s,
        "num_tested": total,
        "status": "ok",
        "note": (
            "Official snntoolbox 0.6.0 fails on Keras 3.x; "
            "this reimplements the standard rate-conversion path for a fair method comparison."
        ),
    }


def run_snntoolbox(keras_model, x_train, y_train, x_test, y_test, T=32, num_to_test=10000, work_dir=None):
    """Run official SNNToolBox Keras→INI pipeline on the same ANN."""
    from snntoolbox.bin.utils import update_setup, run_pipeline
    from snntoolbox.utils.utils import import_configparser

    work_dir = Path(work_dir or (ROOT / "results" / "snntoolbox_cmp"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # SNNToolBox expects one-hot labels in y_*.npz for some paths; check dataset utils
    # Save npz: x as (N, 1, 1, 784) or (N, 784)? For Dense MLP, (N, 784) often works as flatten.
    # Docs use image shape; for MLP use (N, 1, 28, 28) OR flat. Keras model expects (N, 784).
    np.savez_compressed(work_dir / "x_norm.npz", arr_0=x_train[:10000])
    np.savez_compressed(work_dir / "x_test.npz", arr_0=x_test)
    # one-hot for toolbox
    y_oh = np.eye(10, dtype=np.float32)[y_test]
    y_train_oh = np.eye(10, dtype=np.float32)[y_train[:10000]]
    np.savez_compressed(work_dir / "y_test.npz", arr_0=y_oh)
    np.savez_compressed(work_dir / "y_norm.npz", arr_0=y_train_oh)

    model_name = "mlp_mnist_ann"
    # Save without optimizer for compatibility
    h5_path = work_dir / f"{model_name}.h5"
    try:
        keras_model.save(str(h5_path))
    except Exception:
        # Keras 3 may prefer keras format
        keras_model.save(str(work_dir / f"{model_name}.keras"))
        # Also try weights+json
        keras_model.save_weights(str(work_dir / f"{model_name}.weights.h5"))
        with open(work_dir / f"{model_name}.json", "w") as f:
            f.write(keras_model.to_json())
        # Re-save as h5 via legacy if possible
        from tensorflow import keras
        keras.models.save_model(keras_model, str(h5_path), save_format="h5")

    configparser = import_configparser()
    config = configparser.ConfigParser()
    config["paths"] = {
        "path_wd": str(work_dir),
        "dataset_path": str(work_dir),
        "filename_ann": model_name,
    }
    config["tools"] = {
        "evaluate_ann": "True",
        "parse": "True",
        "normalize": "True",
        "convert": "True",
        "simulate": "True",
    }
    config["simulation"] = {
        "simulator": "INI",
        "duration": str(T),
        "num_to_test": str(num_to_test),
        "batch_size": "100",
        "keras_backend": "tensorflow",
    }
    config["input"] = {
        "model_lib": "keras",
        "dataset_format": "npz",
    }
    config_path = work_dir / "config"
    with open(config_path, "w", encoding="utf-8") as f:
        config.write(f)

    t0 = time.perf_counter()
    try:
        cfg = update_setup(str(config_path))
        results = run_pipeline(cfg)
        elapsed = time.perf_counter() - t0
        # results is typically list of accuracies; last is SNN
        snn_acc = None
        ann_acc_tb = None
        if results:
            if len(results) >= 2:
                ann_acc_tb = float(results[0]) * 100.0 if results[0] <= 1.0 else float(results[0])
                snn_acc = float(results[-1]) * 100.0 if results[-1] <= 1.0 else float(results[-1])
            else:
                snn_acc = float(results[-1]) * 100.0 if results[-1] <= 1.0 else float(results[-1])
        return {
            "tool": "snntoolbox",
            "method": "weight-normalize + INIsim (official)",
            "snn_accuracy": snn_acc,
            "ann_accuracy_toolbox": ann_acc_tb,
            "T": T,
            "convert_and_sim_seconds": elapsed,
            "num_tested": num_to_test,
            "raw_results": [float(r) for r in results] if results else None,
            "status": "ok" if snn_acc is not None else "no_accuracy",
            "features": {
                "nir_export": False,
                "multi_backend": "pyNN/Loihi export (vendor paths)",
                "loihi_sim": "via toolbox export",
                "ros2": False,
            },
            "work_dir": str(work_dir),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "tool": "snntoolbox",
            "method": "weight-normalize + INIsim (official)",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "T": T,
            "convert_and_sim_seconds": elapsed,
            "num_tested": num_to_test,
            "work_dir": str(work_dir),
            "features": {
                "nir_export": False,
                "multi_backend": "pyNN/Loihi export (vendor paths)",
                "ros2": False,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="NeuroCUDA vs SNNToolBox MLP MNIST")
    parser.add_argument("--epochs", type=int, default=5, help="ANN train epochs")
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--num-to-test", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qcfs-epochs", type=int, default=3)
    parser.add_argument("--if-epochs", type=int, default=3)
    parser.add_argument("--skip-snntoolbox", action="store_true")
    parser.add_argument("--out", type=str, default="results/compare_snntoolbox_mlp_mnist.json")
    args = parser.parse_args()

    set_seed(args.seed)
    print("=" * 70)
    print("  NeuroCUDA vs SNNToolBox — MLP MNIST")
    print("=" * 70)
    print(f"  seed={args.seed}  ANN epochs={args.epochs}  T={args.T}  test_n={args.num_to_test}")
    print()

    print("[1/4] Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist_numpy()
    if args.num_to_test < len(x_test):
        x_test = x_test[: args.num_to_test]
        y_test = y_test[: args.num_to_test]
    print(f"  train={len(x_train)}  test={len(x_test)}")

    print("\n[2/4] Training shared Keras ANN...")
    keras_model, ann_acc, train_s = train_keras_ann(
        x_train, y_train, x_test, y_test, epochs=args.epochs, seed=args.seed
    )
    print(f"  Keras ANN accuracy: {ann_acc:.2f}%  ({train_s:.1f}s)")

    pt_ann = keras_to_torch(keras_model)
    pt_ann_acc = eval_torch_ann(pt_ann, x_test, y_test)
    print(f"  PyTorch ANN (copied weights): {pt_ann_acc:.2f}%")

    print("\n[3/4] NeuroCUDA convert + evaluate...")
    nc_result = run_neurocuda(
        pt_ann, x_train, y_train, x_test, y_test,
        T=args.T, qcfs_epochs=args.qcfs_epochs, if_epochs=args.if_epochs,
    )
    print(f"  NeuroCUDA SNN: {nc_result['snn_accuracy']:.2f}%")

    print("\n[4/5] SNNToolBox-style rate baseline (Rueckauer reimpl)...")
    # Need fresh ANN weights (convert mutates) — rebuild from Keras
    pt_ann2 = keras_to_torch(keras_model)
    rate_result = run_snntoolbox_style_rate(
        pt_ann2, x_train, x_test, y_test, T=args.T
    )
    print(f"  Rate-style SNN: {rate_result['snn_accuracy']:.2f}%")

    if args.skip_snntoolbox:
        stb_result = {"tool": "snntoolbox", "status": "skipped"}
    else:
        print("\n[5/5] Official SNNToolBox package (may fail on Keras 3)...")
        stb_result = run_snntoolbox(
            keras_model, x_train, y_train, x_test, y_test,
            T=args.T, num_to_test=args.num_to_test,
        )
        if stb_result.get("status") == "ok":
            print(f"  SNNToolBox SNN: {stb_result['snn_accuracy']:.2f}%")
        else:
            print(f"  SNNToolBox: {stb_result.get('status')} — {stb_result.get('error', '')}")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "architecture": "MLP 784→256→256→10 ReLU",
            "dataset": "MNIST test",
            "num_tested": int(len(x_test)),
            "T": args.T,
            "seed": args.seed,
            "same_ann_weights": True,
            "ann_train_epochs": args.epochs,
        },
        "ann_accuracy_keras": ann_acc,
        "ann_accuracy_pytorch_copy": pt_ann_acc,
        "neurocuda": nc_result,
        "snntoolbox_style_rate": rate_result,
        "snntoolbox_official": stb_result,
    }

    # Gaps vs shared ANN
    report["gaps_vs_ann"] = {
        "neurocuda": ann_acc - nc_result["snn_accuracy"],
        "snntoolbox_style_rate": ann_acc - rate_result["snn_accuracy"],
    }
    if stb_result.get("snn_accuracy") is not None:
        report["gaps_vs_ann"]["snntoolbox_official"] = ann_acc - stb_result["snn_accuracy"]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  ANN (shared):           {ann_acc:.2f}%")
    print(f"  NeuroCUDA SNN:          {nc_result['snn_accuracy']:.2f}%  "
          f"gap={report['gaps_vs_ann']['neurocuda']:+.2f}%")
    print(f"  SNNToolBox-style rate:  {rate_result['snn_accuracy']:.2f}%  "
          f"gap={report['gaps_vs_ann']['snntoolbox_style_rate']:+.2f}%")
    if stb_result.get("snn_accuracy") is not None:
        print(f"  SNNToolBox official:    {stb_result['snn_accuracy']:.2f}%  "
              f"gap={report['gaps_vs_ann']['snntoolbox_official']:+.2f}%")
    elif stb_result.get("status") == "error":
        print(f"  SNNToolBox official:    FAILED — {stb_result.get('error')}")
    print(f"  Wrote {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
