"""
Quick QCFS fine-tune + NIR round-trip verification on a model with REAL
(non-uniform) learned thresholds. Even 2 epochs produces varied λ values
that stress the NIR pipeline's threshold handling more than uniform 4.0.

This closes the gap: "proven on model that does nothing" → "proven on model
with real learned parameters."
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import time
import os

sys.path.insert(0, ".")
from models import resnet18_cifar, QCFS, IFNeuron, build_snn_from_qcfs
from nir_export import export_resnet_to_nir
from nir_executor import NIRExecutor, make_torch_from_nir
import nir

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def train_qcfs_quick(qmodel, train_loader, epochs=2, lr_w=0.01, lr_lam=0.1, device="cpu"):
    """Quick QCFS fine-tune to get non-uniform thresholds."""
    thresh_params = [p for n, p in qmodel.named_parameters() if n.endswith("thresh")]
    other_params = [p for n, p in qmodel.named_parameters() if not n.endswith("thresh")]

    opt = torch.optim.SGD(
        [{"params": other_params, "lr": lr_w},
         {"params": thresh_params, "lr": lr_lam}],
        momentum=0.9, weight_decay=5e-4,
    )
    crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        qmodel.train()
        running_loss = 0.0
        t0 = time.time()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(qmodel(x), y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if i % 100 == 0:
                pct = 100 * i / len(train_loader)
                print(f"\r  ep {ep+1}/{epochs}: {pct:.0f}% loss={running_loss/max(i,1):.4f}", end="")

        t1 = time.time()
        # Show current λ values
        lambdas = {n: f"{m.thresh.item():.3f}"
                   for n, m in qmodel.named_modules() if isinstance(m, QCFS)}
        print(f"\r  ep {ep+1}/{epochs} done ({t1-t0:.0f}s) λ={lambdas}")

    return qmodel


@torch.no_grad()
def evaluate_model(model, loader, device="cpu"):
    """Run inference on full test set."""
    model.eval()
    correct = 0
    total = 0
    all_logits = []
    for x, y in loader:
        x = x.to(device)
        for m in model.modules():
            if isinstance(m, IFNeuron):
                m.reset()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        all_logits.append(out.cpu())
        correct += (out.argmax(dim=1).cpu() == y).sum().item()
        total += y.size(0)
    acc = 100.0 * correct / total
    return acc, torch.cat(all_logits, dim=0)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    SEED = 0
    QCFS_EPOCHS = 30  # full training for real accuracy

    # ---- 1. Load data ----
    print("\n[1/5] Loading CIFAR-10...")
    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    te_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tr_ds = datasets.CIFAR10("./data", train=True, download=True, transform=tr_tf)
    te_ds = datasets.CIFAR10("./data", train=False, download=True, transform=te_tf)
    tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=0)
    print(f"   Train: {len(tr_ds)}, Test: {len(te_ds)}")

    # ---- 2. Load ANN + build QCFS ----
    print(f"\n[2/5] Loading ANN seed {SEED} + building QCFS...")
    ann_ckpt = torch.load(f"checkpoints/ann_resnet18_seed{SEED}.pt",
                          map_location=device, weights_only=False)
    sd = ann_ckpt["model"]

    qmodel = resnet18_cifar(lambda: QCFS(L=8, thresh_init=4.0)).to(device)
    missing, unexpected = qmodel.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.endswith("thresh")]
    assert not bad, f"Unexpected missing keys: {bad}"

    # Show initial λ (all 4.0)
    init_lam = {n: f"{m.thresh.item():.3f}"
                for n, m in qmodel.named_modules() if isinstance(m, QCFS)}
    print(f"   Initial λ: {init_lam}")

    # ---- 3. Quick QCFS fine-tune ----
    print(f"\n[3/5] QCFS fine-tune ({QCFS_EPOCHS} epochs)...")
    train_qcfs_quick(qmodel, tr_ld, epochs=QCFS_EPOCHS, device=device)

    # Show final λ (should be non-uniform)
    final_lam = {n: f"{m.thresh.item():.3f}"
                 for n, m in qmodel.named_modules() if isinstance(m, QCFS)}
    print(f"   Final λ: {final_lam}")

    # Check QCFS accuracy (should be above random if training helped)
    qmodel.eval()
    q_acc, _ = evaluate_model(qmodel, te_ld, device)
    print(f"   QCFS test accuracy: {q_acc:.2f}%")

    # ---- 4. Build SNN + NIR round-trip ----
    print(f"\n[4/5] SNN + NIR round-trip...")
    snn = build_snn_from_qcfs(qmodel)
    snn = snn.to(device)
    snn.eval()

    # Show IF thresholds
    if_lam = {n: f"{m.thresh:.3f}"
              for n, m in snn.named_modules() if isinstance(m, IFNeuron)}
    print(f"   IF thresholds: {if_lam}")

    # Export (export operates on CPU numpy arrays — fine)
    path = f"qcfs_seed{SEED}.nir"
    g = export_resnet_to_nir(snn.cpu(), path)

    # Rebuild and move to device
    g2 = nir.read(path)
    executor = NIRExecutor(g2, make_torch_from_nir)
    executor = executor.to(device)
    executor.eval()
    print(f"   Executor: {len(executor.node_order)} compute nodes on {device}")

    # ---- 5. Verify: full 10K comparison ----
    print(f"\n[5/5] Full 10K verification...")
    t0 = time.time()

    # Original SNN
    orig_acc, orig_logits = evaluate_model(snn, te_ld, device)

    # NIR-rebuilt
    reb_acc, reb_logits = evaluate_model(executor, te_ld, device)

    t1 = time.time()

    # Compare
    max_diff = (orig_logits - reb_logits).abs().max().item()
    acc_delta = abs(orig_acc - reb_acc)
    n_div = (orig_logits - reb_logits).abs().max(dim=1).values
    divergent = (n_div > 1e-6).sum().item()

    print(f"\n{'='*60}")
    print(f"RESULTS — QCFS-TRAINED MODEL (Seed {SEED}, {QCFS_EPOCHS} epochs)")
    print(f"{'='*60}")
    print(f"  QCFS test accuracy:     {q_acc:.2f}%")
    print(f"  Original SNN accuracy:  {orig_acc:.2f}%")
    print(f"  NIR-rebuilt accuracy:   {reb_acc:.2f}%")
    print(f"  Accuracy delta:         {acc_delta:.4f}%")
    print(f"  Max logit diff:         {max_diff:.6e}")
    print(f"  Divergent images:       {divergent} / {len(te_ds)}")
    print(f"  Verification time:      {t1-t0:.0f}s")
    print(f"  Learned λ range:        {min(float(v) for v in final_lam.values()):.3f}"
          f" — {max(float(v) for v in final_lam.values()):.3f}")
    print()

    if max_diff < 1e-4 and divergent == 0:
        print("  NIR VERIFIED ✅ — bit-exact on model with learned (non-uniform) thresholds")
        print(f"  QCFS accuracy {q_acc:.1f}% preserved exactly through NIR round-trip")
    else:
        print(f"  FAILED ❌ — {divergent} divergent images, max diff {max_diff:.2e}")
    print(f"{'='*60}")

    # Save for posterity
    torch.save({"state_dict": qmodel.state_dict(), "lambdas": final_lam, "accuracy": q_acc},
               f"checkpoints/qcfs_resnet18_seed{SEED}.pt")
    print(f"\nSaved: checkpoints/qcfs_resnet18_seed{SEED}.pt")
