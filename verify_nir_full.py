"""
Full NIR verification: all seeds, full 10K CIFAR-10 test set.
Compares original SNN vs NIR-reimported accuracy.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, ".")
from models import resnet18_cifar, QCFS, IFNeuron, build_snn_from_qcfs
from nir_export import export_resnet_to_nir
from nir_executor import NIRExecutor, make_torch_from_nir
import nir


@torch.no_grad()
def evaluate_model(model, loader, device="cpu"):
    """Run full test set inference. Returns (accuracy, all_logits)."""
    model.eval()
    correct = 0
    total = 0
    all_logits = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        # Reset IF state per batch
        for m in model.modules():
            if isinstance(m, IFNeuron):
                m.reset()

        out = model(x)
        if isinstance(out, tuple):
            out = out[0]

        all_logits.append(out.cpu())
        all_labels.append(y)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = 100.0 * correct / total
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return acc, all_logits, all_labels


def verify_seed(seed, device="cpu"):
    """Full verification for one seed."""
    print(f"\n{'='*60}")
    print(f"SEED {seed} — FULL NIR VERIFICATION")
    print(f"{'='*60}")

    # 1. Load checkpoint
    print(f"\n[1/5] Loading seed {seed} checkpoint...")
    ckpt = torch.load(f"checkpoints/ann_resnet18_seed{seed}.pt",
                       map_location=device, weights_only=False)
    sd = ckpt["model"]

    # 2. Build model
    print("[2/5] Building SNN model...")
    q = resnet18_cifar(lambda: QCFS(L=8))
    q.load_state_dict(sd, strict=False)  # QCFS thresholds use defaults
    snn = build_snn_from_qcfs(q)
    snn = snn.to(device)
    snn.eval()
    params = sum(p.numel() for p in snn.parameters())
    print(f"      {params:,} parameters")

    # 3. Export to NIR
    print("[3/5] Exporting to NIR...")
    path = f"resnet_seed{seed}.nir"
    export_resnet_to_nir(snn, path)

    # 4. Rebuild from NIR
    print("[4/5] Rebuilding from NIR...")
    g = nir.read(path)
    executor = NIRExecutor(g, make_torch_from_nir)
    executor = executor.to(device)
    executor.eval()
    print(f"      {len(executor.node_order)} execution nodes")

    # 5. Load CIFAR-10 test set
    print("[5/5] Loading CIFAR-10 test set...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    testset = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform)
    loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    print(f"      {len(testset)} images, {len(loader)} batches")

    # 6. Evaluate original
    print("\n=== Evaluating ORIGINAL SNN ===")
    t0 = time.time()
    orig_acc, orig_logits, labels = evaluate_model(snn, loader, device)
    t1 = time.time()
    print(f"      Accuracy: {orig_acc:.2f}%")
    print(f"      Time: {t1-t0:.1f}s")

    # 7. Evaluate NIR-rebuilt
    print("\n=== Evaluating NIR-REBUILT ===")
    t0 = time.time()
    rebuilt_acc, rebuilt_logits, _ = evaluate_model(executor, loader, device)
    t1 = time.time()
    print(f"      Accuracy: {rebuilt_acc:.2f}%")
    print(f"      Time: {t1-t0:.1f}s")

    # 8. Compare
    max_diff = (orig_logits - rebuilt_logits).abs().max().item()
    acc_delta = abs(orig_acc - rebuilt_acc)
    per_image_diffs = (orig_logits - rebuilt_logits).abs().max(dim=1).values
    num_divergent = (per_image_diffs > 1e-6).sum().item()

    print(f"\n=== COMPARISON ===")
    print(f"      Original accuracy:  {orig_acc:.4f}%")
    print(f"      Rebuilt accuracy:   {rebuilt_acc:.4f}%")
    print(f"      Accuracy delta:     {acc_delta:.6f}%")
    print(f"      Max logit diff:     {max_diff:.6e}")
    print(f"      Divergent images:   {num_divergent} / {len(testset)}")

    passed = max_diff < 1e-4 and num_divergent == 0
    if passed:
        print(f"\n      SEED {seed}: PASS ✅ (bit-exact on all {len(testset)} images)")
    else:
        print(f"\n      SEED {seed}: FAIL ❌ ({num_divergent} divergent images, max diff {max_diff:.2e})")

    return {
        "seed": seed,
        "orig_acc": orig_acc,
        "rebuilt_acc": rebuilt_acc,
        "acc_delta": acc_delta,
        "max_diff": max_diff,
        "num_divergent": num_divergent,
        "passed": passed,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Torch: {torch.__version__}")

    results = []
    for seed in [0, 1, 2]:
        r = verify_seed(seed, device)
        results.append(r)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL NIR VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"{'Seed':<6} {'Orig%':<10} {'Rebuilt%':<10} {'ΔAcc%':<12} {'MaxDiff':<12} {'Status'}")
    print("-" * 70)
    all_pass = True
    for r in results:
        status = "PASS ✅" if r["passed"] else "FAIL ❌"
        print(f"{r['seed']:<6} {r['orig_acc']:<10.4f} {r['rebuilt_acc']:<10.4f} "
              f"{r['acc_delta']:<12.6f} {r['max_diff']:<12.2e} {status}")
        if not r["passed"]:
            all_pass = False

    print("-" * 70)
    if all_pass:
        orig_mean = np.mean([r["orig_acc"] for r in results])
        orig_std = np.std([r["orig_acc"] for r in results])
        reb_mean = np.mean([r["rebuilt_acc"] for r in results])
        reb_std = np.std([r["rebuilt_acc"] for r in results])
        print(f"\nMean accuracy:  original={orig_mean:.2f}% ± {orig_std:.2f}")
        print(f"                rebuilt ={reb_mean:.2f}% ± {reb_std:.2f}")
        print(f"                delta   ={abs(orig_mean-reb_mean):.6f}%")
        print(f"\n🎉 NIR EXPORT — FULLY PROVEN")
        print(f"   All 3 seeds, full 10K test set, bit-exact match")
        print(f"   RESNET: WRITE ✅ READ ✅ EXECUTE ✅ ACCURACY ✅")
    else:
        print(f"\n⚠️  SOME SEEDS FAILED — investigation needed")
    print("=" * 70)
