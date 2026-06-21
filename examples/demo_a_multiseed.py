"""
NMNIST Multi-Seed Conversion — QCFS → IF → BPTT FT
====================================================
Runs the full ANN→SNN conversion pipeline for 3 seeds,
measuring accuracy, gap, and sparsity.

Pipeline per seed:
  1. Train ANN (5 epochs, quick — just for calibration baseline)
     OR load pretrained ANN checkpoint
  2. QCFS calibrate (per-channel thresholds, 3 epochs)
  3. BN fold + IF replace
  4. BPTT fine-tune (3 epochs)
  5. Measure: accuracy, gap, sparsity

Usage: python examples/demo_a_multiseed.py [--seeds 0 1 2]
"""
import sys, os, time, copy, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurocuda import convert, measure_sparsity
from models import QCFS, IFNeuron, reset_spiking, build_snn_from_qcfs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Model
# ===========================================================================

class NMNISTCNN(nn.Module):
    """4D-native CNN for N-MNIST (B, C, H, W) — used with temporal wrapper."""
    def __init__(self, act_factory=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32); self.act1 = act_factory(); self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64); self.act2 = act_factory(); self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128); self.act3 = act_factory(); self.pool3 = nn.AvgPool2d(2)
        self.flatten = nn.Flatten(); self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        return self.fc(self.flatten(x))


class TemporalWrapper(nn.Module):
    """Wrap 4D-native model for 5D (B,T,C,H,W) input. Returns (B, num_classes)."""
    def __init__(self, model_4d):
        super().__init__()
        self.model_4d = model_4d

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        out = self.model_4d(x)
        return out.reshape(B, T, -1).mean(dim=1)


# ===========================================================================
# BN Folding
# ===========================================================================

def fold_conv_bn_generic(model):
    for parent_mod in list(model.modules()):
        children = list(parent_mod.named_children())
        for i in range(len(children) - 1):
            c_mod, n_mod = children[i][1], children[i + 1][1]
            if isinstance(c_mod, nn.Conv2d) and isinstance(n_mod, nn.BatchNorm2d):
                conv, bn = c_mod, n_mod
                w = conv.weight
                gamma, beta = bn.weight, bn.bias
                rm, rv, eps = bn.running_mean, bn.running_var, bn.eps
                std = torch.sqrt(rv + eps)
                scale = (gamma / std) if gamma is not None else (1.0 / std)
                fused_w = w * scale.reshape(-1, 1, 1, 1)
                b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=w.device)
                fused_b = (beta + (b_conv - rm) * gamma / std) if (gamma is not None and beta is not None) else (b_conv - rm / std)
                with torch.no_grad():
                    conv.weight.copy_(fused_w)
                    if conv.bias is not None: conv.bias.copy_(fused_b)
                    else: conv.bias = nn.Parameter(fused_b)
                setattr(parent_mod, children[i + 1][0], nn.Identity())
    return model


# ===========================================================================
# Accuracy measurement
# ===========================================================================

def measure_accuracy(model, dataloader, is_spiking=False, T=16):
    """Measure classification accuracy on full dataloader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            if is_spiking:
                # Per-frame forward for 5D data
                B, T_data = data.size(0), min(T, data.size(1))
                reset_spiking(model)
                out_sum = torch.zeros(B, 10, device=device)
                for t in range(T_data):
                    out_sum += model(data[:, t:t+1, :, :, :])
                pred = out_sum.argmax(1)
            else:
                pred = model(data).argmax(1)
            correct += (pred == target).sum().item()
            total += data.size(0)
    return 100.0 * correct / total


# ===========================================================================
# Run one seed
# ===========================================================================

def run_seed(seed, train_loader, test_loader):
    print(f"\n{'='*60}")
    print(f"NMNIST Seed {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    # --- 1. ANN baseline ---
    print("\n[1/4] ANN baseline...")
    ann_model = TemporalWrapper(NMNISTCNN(act_factory=nn.ReLU)).to(device)

    # Quick train or load checkpoint
    ckpt_path = "./checkpoints/demo_a_ann_best.pt"
    inner_model = ann_model.model_4d

    if os.path.exists(ckpt_path):
        inner_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  Loaded pretrained ANN from {ckpt_path}")
    else:
        print("  Training ANN (5 epochs)...")
        inner_model.train()
        opt = torch.optim.AdamW(inner_model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        for ep in range(5):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                loss = crit(ann_model(data), target)
                loss.backward(); opt.step()
        inner_model.eval()
        torch.save(inner_model.state_dict(), ckpt_path)
        print(f"  Trained quick ANN, saved to {ckpt_path}")

    ann_acc = measure_accuracy(ann_model, test_loader, is_spiking=False)
    print(f"  ANN accuracy: {ann_acc:.2f}%")

    # --- 2. Convert via neurocuda.convert() ---
    print("\n[2/3] neurocuda.convert() — CS-QCFS + IF + BPTT...")
    t0 = time.time()

    # Wrap the 4D-native CNN for 5D temporal input
    snn_inner = TemporalWrapper(inner_model).to(device)

    snn_model, stats = convert(
        snn_inner,
        train_loader,
        test_loader=test_loader,
        qcfs_epochs=3,
        if_epochs=3,
        strategy="qcfs_if_ft",
        channel_wise=True,
        device=device,
        verbose=False
    )

    qcfs_acc = stats["qcfs_accuracy"]
    if_acc = stats["if_accuracy"]

    conv_time = time.time() - t0
    print(f"  QCFS accuracy: {qcfs_acc:.2f}%")
    print(f"  IF accuracy:   {if_acc:.2f}%")
    print(f"  Gap:           {ann_acc - if_acc:.2f}%")
    print(f"  Conversion time: {conv_time/60:.1f} min")

    # --- 3. Sparsity ---
    print("\n[3/3] Sparsity measurement...")
    sparsity, nonzero, total_acts, layer_data = measure_sparsity(
        snn_model, test_loader, device=device, max_batches=20
    )
    print(f"  Sparsity: {sparsity:.2f}%")
    for name, d in sorted(layer_data.items()):
        ls = 100.0 * (1.0 - d["nonzero"] / max(d["total"], 1))
        print(f"    {name}: {ls:.1f}% sparse")

    return {
        "seed": seed,
        "ann_acc": ann_acc,
        "qcfs_acc": qcfs_acc,
        "if_acc": if_acc,
        "gap": ann_acc - if_acc,
        "sparsity": sparsity,
    }


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n_test", type=int, default=2000,
                       help="Number of test samples (default: 2000)")
    parser.add_argument("--n_train", type=int, default=20000,
                       help="Number of train/calibration samples (default: 20000)")
    args = parser.parse_args()

    print(f"Device: {device}")
    print(f"NMNIST Multi-Seed Conversion ({len(args.seeds)} seeds)")
    print(f"Seeds: {args.seeds}")
    print(f"Test samples: {args.n_test}, Train samples: {args.n_train}")

    # Load data once
    print("\nLoading NMNIST data...")
    try:
        test_data = torch.load("./data/nmnist_test.pt", map_location="cpu", weights_only=False)
        test_loader = DataLoader(
            TensorDataset(test_data["data"][:args.n_test], test_data["targets"][:args.n_test]),
            batch_size=128)

        train_data = torch.load("./data/nmnist_train.pt", map_location="cpu", weights_only=False)
        n_train = min(args.n_train, len(train_data["data"]))
        train_loader = DataLoader(
            TensorDataset(train_data["data"][:n_train], train_data["targets"][:n_train]),
            batch_size=128, shuffle=True)
        print(f"  Test: {args.n_test} frames | Train (calib): {n_train} frames")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("  Run examples/prep_nmnist.py first.")
        sys.exit(1)

    t_start = time.time()

    all_results = []
    for seed in args.seeds:
        result = run_seed(seed, train_loader, test_loader)
        all_results.append(result)

    # =========================================================================
    # Aggregate
    # =========================================================================
    print(f"\n{'='*60}")
    print("NMNIST MULTI-SEED SUMMARY")
    print(f"{'='*60}")

    ann_accs = [r["ann_acc"] for r in all_results]
    if_accs = [r["if_acc"] for r in all_results]
    gaps = [r["gap"] for r in all_results]
    sparsities = [r["sparsity"] for r in all_results]

    print(f"\n  {'Seed':<8s} {'ANN':>8s} {'IF':>8s} {'Gap':>8s} {'Sparsity':>10s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for r in all_results:
        print(f"  {r['seed']:<8d} {r['ann_acc']:>7.2f}% {r['if_acc']:>7.2f}% "
              f"{r['gap']:>7.2f}% {r['sparsity']:>9.1f}%")

    print(f"\n  AGGREGATE (mean ± std over {len(args.seeds)} seeds):")
    print(f"  ANN:       {np.mean(ann_accs):.2f}% ± {np.std(ann_accs):.2f}%")
    print(f"  IF (SNN):  {np.mean(if_accs):.2f}% ± {np.std(if_accs):.2f}%")
    print(f"  Gap:       {np.mean(gaps):.2f}% ± {np.std(gaps):.2f}%")
    print(f"  Sparsity:  {np.mean(sparsities):.2f}% ± {np.std(sparsities):.2f}%")
    print(f"  Total time: {(time.time() - t_start) / 60:.1f} min")
    print(f"{'='*60}")
