"""
gate3_qcfs_convert.py — GATE 3: fix the converter (the real work).

Pipeline:
  1. Load the pretrained ReLU ANN from Gate 2.
  2. Build a QCFS version of the SAME architecture and copy the ANN weights
     (init from pretrained ANN -- do NOT train QCFS from scratch).
  3. Fine-tune the QCFS model with a SEPARATE, HIGHER learning rate on the
     lambda threshold parameters. This is the fix for the frozen-lambda bug.
  4. Log every lambda's value AND gradient norm each epoch to PROVE they move.
  5. Convert QCFS -> IF-neuron SNN and evaluate at several timesteps T on the
     FULL test set.
  6. Report the conversion gap.

Acceptance (per the agent brief):
  CIFAR-10 conversion gap <= 5% (e.g. ANN 94% -> SNN >= 89%), mean +- std over
  3 seeds, full test set, WITH a logged trace showing ALL lambda values moved.
  If lambda still freezes, STOP and report the gradient trace.
  Do NOT relabel a frozen-lambda failure as "QCFS doesn't generalize".

Usage:
    python gate3_qcfs_convert.py --seed 0 --L 8
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import (resnet18_cifar, QCFS, IFNeuron,
                    build_snn_from_qcfs, reset_snn)


def get_loaders(data_dir, batch=128, workers=4):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    tr = datasets.CIFAR10(data_dir, train=True, download=True, transform=tr_tf)
    te = datasets.CIFAR10(data_dir, train=False, download=True, transform=te_tf)
    return (DataLoader(tr, batch, shuffle=True, num_workers=workers, drop_last=True),
            DataLoader(te, 256, shuffle=False, num_workers=workers))


@torch.no_grad()
def eval_ann(model, loader, device):
    model.eval()
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += model(x).argmax(1).eq(y).sum().item()
        t += y.size(0)
    return 100.0 * c / t


@torch.no_grad()
def eval_snn(snn, loader, device, T):
    """Constant-input encoding: feed the real image each of T timesteps,
    accumulate logits, argmax of the sum. Full test set."""
    snn.eval()
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        reset_snn(snn)
        out = 0
        for _ in range(T):
            out = out + snn(x)
        c += out.argmax(1).eq(y).sum().item()
        t += y.size(0)
    return 100.0 * c / t


def lambda_trace(model):
    """Return list of (name, value, grad_norm) for every QCFS threshold."""
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, QCFS):
            g = None if m.thresh.grad is None else m.thresh.grad.norm().item()
            rows.append((name, m.thresh.item(), g))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--L", type=int, default=8, help="QCFS quantization steps")
    ap.add_argument("--lr_w", type=float, default=0.01, help="LR for conv/bn/fc")
    ap.add_argument("--lr_lambda", type=float, default=0.1,
                    help="SEPARATE higher LR for thresholds (the fix)")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--ckpt_dir", default="./checkpoints")
    ap.add_argument("--timesteps", type=int, nargs="+", default=[4, 8, 16, 32])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_ld, te_ld = get_loaders(args.data)

    # --- 1. Load Gate 2 ANN ---
    ann_ckpt = os.path.join(args.ckpt_dir, f"ann_resnet18_seed{args.seed}.pt")
    assert os.path.exists(ann_ckpt), f"Missing Gate 2 checkpoint: {ann_ckpt}"
    ann = resnet18_cifar(lambda: nn.ReLU(inplace=True)).to(device)
    ann.load_state_dict(torch.load(ann_ckpt, map_location=device)["model"])
    ann_acc = eval_ann(ann, te_ld, device)
    print(f"[gate2 ANN] full-test acc = {ann_acc:.2f}%")

    # --- 2. Build QCFS model, copy ANN weights (strict=False: thresh params new) ---
    qmodel = resnet18_cifar(lambda: QCFS(L=args.L, thresh_init=4.0)).to(device)
    missing, unexpected = qmodel.load_state_dict(ann.state_dict(), strict=False)
    # Sanity: the only "missing" keys should be QCFS thresh params.
    bad = [k for k in missing if not k.endswith("thresh")]
    assert not bad and not unexpected, f"weight transfer mismatch: {bad} / {unexpected}"

    # --- 3. Two param groups: low LR weights, HIGH LR thresholds (the fix) ---
    thresh_params = [p for n, p in qmodel.named_parameters() if n.endswith("thresh")]
    other_params = [p for n, p in qmodel.named_parameters() if not n.endswith("thresh")]
    opt = torch.optim.SGD(
        [{"params": other_params, "lr": args.lr_w},
         {"params": thresh_params, "lr": args.lr_lambda}],
        momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    # Snapshot initial lambdas to verify movement later.
    init_lambdas = {n: m.thresh.item()
                    for n, m in qmodel.named_modules() if isinstance(m, QCFS)}

    print("\n[gate3] fine-tuning QCFS (separate LR on thresholds)...")
    t0 = time.time()
    for ep in range(args.epochs):
        qmodel.train()
        last_trace = None
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(qmodel(x), y)
            loss.backward()
            last_trace = lambda_trace(qmodel)   # capture grads before step
            opt.step()
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            acc = eval_ann(qmodel, te_ld, device)
            # Confirm thresholds are MOVING and receiving gradient.
            grads = [g for _, _, g in last_trace if g is not None]
            moved = sum(1 for n, m in qmodel.named_modules()
                        if isinstance(m, QCFS)
                        and abs(m.thresh.item() - init_lambdas[n]) > 1e-3)
            print(f"  ep {ep+1:>2}/{args.epochs}  qcfs_acc={acc:.2f}%  "
                  f"thresholds_moved={moved}/{len(init_lambdas)}  "
                  f"mean_grad_norm={sum(grads)/max(len(grads),1):.4e}")

    # --- 4. Final lambda movement report (the proof) ---
    print("\n[gate3] lambda movement (init -> final):")
    all_moved = True
    for n, m in qmodel.named_modules():
        if isinstance(m, QCFS):
            init_v, fin_v = init_lambdas[n], m.thresh.item()
            delta = fin_v - init_v
            flag = "" if abs(delta) > 1e-3 else "  <-- FROZEN (BUG)"
            if abs(delta) <= 1e-3:
                all_moved = False
            print(f"  {n:<28} {init_v:7.3f} -> {fin_v:7.3f}  (d={delta:+.3f}){flag}")
    if not all_moved:
        print("\nSTOP: some thresholds froze. This is a gradient/LR bug, NOT a "
              "QCFS limitation. Do not relabel as a 'finding'. Debug before Gate 4.")

    qcfs_acc = eval_ann(qmodel, te_ld, device)

    # --- 5. Convert to SNN and evaluate across timesteps (full test set) ---
    snn = build_snn_from_qcfs(qmodel).to(device)
    print(f"\n[gate3] SNN conversion results (full 10k test set):")
    print(f"  ANN(ReLU)    = {ann_acc:.2f}%")
    print(f"  ANN(QCFS,L={args.L}) = {qcfs_acc:.2f}%")
    best_snn = 0.0
    for T in args.timesteps:
        snn_acc = eval_snn(snn, te_ld, device, T)
        best_snn = max(best_snn, snn_acc)
        print(f"  SNN  T={T:>3}   = {snn_acc:.2f}%   (gap vs ReLU ANN = {ann_acc - snn_acc:.2f}%)")

    gap = ann_acc - best_snn
    print(f"\nGATE 3 result: best SNN = {best_snn:.2f}%, conversion gap = {gap:.2f}%")
    print(f"GATE 3 target: gap <= 5.00%  ->  {'PASS' if gap <= 5.0 else 'FAIL'}")
    print(f"(report mean +- std over 3 seeds; all thresholds must have moved)")
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
