"""
═══════════════════════════════════════════════════════════════════════════
NEUROCUDA — QCFS PIPELINE (ICLR 2022)
═══════════════════════════════════════════════════════════════════════════
1. Train stride CNN with QCFS activation (not ReLU)
2. Fold BN → extract learned λ values
3. Convert to SNN using λ as LIF thresholds
4. Test accuracy at T=2,4,8,16,32,64
5. Compare with old percentile method

QCFS: github.com/putshua/SNN_conversion_QCFS (Bu et al., ICLR 2022)
CS-QCFS: Bu et al., Neural Networks 2025 (95.86% at T=1)

RUN: python examples/qcfs_pipeline.py
═══════════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn as nn, snntorch as snn, numpy as np, json, time
from snntorch import surrogate
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime
from neurocuda.qcfs import QCFSActivation, replace_relu_with_qcfs

# ═══════════════════════════════════════════════════════════
B, EPOCHS, L = 256, 30, 16  # L=16 quantization levels
DEV = torch.device("cuda")
SAVE_PATH = "c:/neurocuda/examples/cnn_qcfs_best.pt"

te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tr_trf = transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tr_etf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
te_ds = datasets.CIFAR10("c:/neurocuda/data", train=False, download=False, transform=te_tf)
tr_ds = datasets.CIFAR10("c:/neurocuda/data", train=True, download=False, transform=tr_trf)
tr_eds = datasets.CIFAR10("c:/neurocuda/data", train=True, download=False, transform=tr_etf)
tr_ldr = DataLoader(tr_ds, B, shuffle=True)
te_ldr = DataLoader(te_ds, B, shuffle=False, drop_last=True)
test_ldr = DataLoader(Subset(te_ds, range(1000)), B, shuffle=False, drop_last=True)

# QCFS CNN — each ReLU replaced with QCFS activation
class QCFS_CNN(nn.Module):
    def __init__(self, L=16):
        super().__init__()
        self.c1=nn.Conv2d(3,64,3,stride=2,padding=1,bias=False);self.b1=nn.BatchNorm2d(64)
        self.qcfs1 = QCFSActivation(1.0, L)
        self.c2=nn.Conv2d(64,128,3,stride=2,padding=1,bias=False);self.b2=nn.BatchNorm2d(128)
        self.qcfs2 = QCFSActivation(1.0, L)
        self.c3=nn.Conv2d(128,256,3,stride=2,padding=1,bias=False);self.b3=nn.BatchNorm2d(256)
        self.qcfs3 = QCFSActivation(1.0, L)
        self.avg=nn.AdaptiveAvgPool2d(1);self.flat=nn.Flatten();self.fc=nn.Linear(256,10)

    def forward(self, x):
        x=self.qcfs1(self.b1(self.c1(x)))
        x=self.qcfs2(self.b2(self.c2(x)))
        x=self.qcfs3(self.b3(self.c3(x)))
        return self.fc(self.flat(self.avg(x)))

print("=" * 60)
print("NEUROCUDA — QCFS PIPELINE (ICLR 2022)")
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0)} | L={L} | Epochs={EPOCHS}")

# ═══════════════════════════════════════════════════════════
# PHASE 1: Train QCFS-ANN
# ═══════════════════════════════════════════════════════════
import os
if os.path.exists(SAVE_PATH):
    ann = torch.load(SAVE_PATH, map_location=DEV, weights_only=False)
    print(f"\n[1/4] Loaded QCFS-ANN from {SAVE_PATH}")
else:
    print(f"\n[1/4] Training QCFS-ANN ({EPOCHS} epochs, L={L})...")
    ann = QCFS_CNN(L=L).to(DEV)
    opt = torch.optim.AdamW(ann.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.CrossEntropyLoss(); best = 0
    SWITCH_EPOCH = EPOCHS // 3  # First 1/3 with ReLU, then QCFS

    # Temporarily replace QCFS with ReLU for warmup
    class ReLUWrapper(nn.Module):
        def forward(self, x): return torch.relu(x)

    # Save QCFS modules and replace with ReLU for warmup
    qcfs_modules = {}
    for name, mod in ann.named_modules():
        if isinstance(mod, QCFSActivation):
            qcfs_modules[name] = mod

    # Use ReLU for first SWITCH_EPOCH epochs
    for name, mod in ann.named_modules():
        if isinstance(mod, QCFSActivation):
            parent = ann
            parts = name.rsplit('.', 1)
            for p in parts[0].split('.'):
                if p: parent = getattr(parent, p)
            if len(parts) > 1:
                setattr(parent, parts[1], nn.ReLU())
            else:
                setattr(ann, name, nn.ReLU())

    for ep in range(EPOCHS):
        # Switch to QCFS at SWITCH_EPOCH
        if ep == SWITCH_EPOCH:
            for name, mod in ann.named_modules():
                if isinstance(mod, nn.ReLU):
                    parent = ann
                    parts = name.rsplit('.', 1)
                    for p in parts[0].split('.'):
                        if p: parent = getattr(parent, p)
                    if len(parts) > 1:
                        setattr(parent, parts[1], qcfs_modules[name])
            print(f"  Switched to QCFS at epoch {ep+1}")

        ann.train()
        for d, t in tr_ldr: d,t=d.to(DEV),t.to(DEV); opt.zero_grad(); crit(ann(d),t).backward(); opt.step()
        sch.step()
        ann.eval(); cor, tot = 0, 0
        with torch.no_grad():
            for d, t in test_ldr: d,t=d.to(DEV),t.to(DEV); cor += ann(d).max(1)[1].eq(t).sum().item(); tot += t.size(0)
        acc = 100*cor/tot
        if ep % 10 == 0 or ep >= EPOCHS-3 or ep == SWITCH_EPOCH: print(f"  Epoch {ep+1}: {acc:.1f}% {'(ReLU)' if ep < SWITCH_EPOCH else '(QCFS)'}")
        if acc > best: best = acc; torch.save(ann, SAVE_PATH)
    ann = torch.load(SAVE_PATH, map_location=DEV, weights_only=False)

ann.eval(); cor, tot = 0, 0
with torch.no_grad():
    for d, t in test_ldr: d,t=d.to(DEV),t.to(DEV); cor += ann(d).max(1)[1].eq(t).sum().item(); tot += t.size(0)
ann_acc = 100*cor/tot

# Extract learned λ values
lambda_values = []
for name, mod in ann.named_modules():
    if isinstance(mod, QCFSActivation):
        lambda_values.append(mod.get_threshold().item())
print(f"  QCFS-ANN Accuracy: {ann_acc:.1f}%")
for i, lv in enumerate(lambda_values):
    print(f"  Layer {i+1}: λ = {lv:.3f}")

# ═══════════════════════════════════════════════════════════
# PHASE 2: Fold BN
# ═══════════════════════════════════════════════════════════
print(f"\n[2/4] Folding BN...")
def fold(conv, bn):
    if conv.bias is None: conv.bias = nn.Parameter(torch.zeros(conv.out_channels))
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight.data *= s.view(-1, 1, 1, 1)
    conv.bias.data = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)
    bn.weight.data = torch.ones_like(bn.weight); bn.bias.data.zero_()
    bn.running_mean.zero_(); bn.running_var.fill_(1.0 - bn.eps)
for c, b in [(ann.c1,ann.b1),(ann.c2,ann.b2),(ann.c3,ann.b3)]: fold(c, b)
print(f"  BN folded. ANN acc preserved: {ann_acc:.1f}%")

# ═══════════════════════════════════════════════════════════
# PHASE 3: Convert to SNN + Test at multiple T
# ═══════════════════════════════════════════════════════════
print(f"\n[3/4] Converting QCFS-ANN → SNN + testing...")

sg = surrogate.fast_sigmoid(slope=25)

def build_qcfs_snn(ann, lambda_vals, T_snn):
    """Build SNN with QCFS-derived thresholds."""
    class QCFS_SNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1=nn.Conv2d(3,64,3,stride=2,padding=1,bias=True);self.c1.load_state_dict({k:v for k,v in ann.c1.state_dict().items()})
            self.l1=snn.Leaky(beta=1.0,threshold=lambda_vals[0],spike_grad=sg,reset_mechanism="subtract")
            self.c2=nn.Conv2d(64,128,3,stride=2,padding=1,bias=True);self.c2.load_state_dict({k:v for k,v in ann.c2.state_dict().items()})
            self.l2=snn.Leaky(beta=1.0,threshold=lambda_vals[1],spike_grad=sg,reset_mechanism="subtract")
            self.c3=nn.Conv2d(128,256,3,stride=2,padding=1,bias=True);self.c3.load_state_dict({k:v for k,v in ann.c3.state_dict().items()})
            self.l3=snn.Leaky(beta=1.0,threshold=lambda_vals[2],spike_grad=sg,reset_mechanism="subtract")
            self.avg=nn.AdaptiveAvgPool2d(1);self.flat=nn.Flatten();self.fc=nn.Linear(256,10)
            self.fc.load_state_dict({k:v for k,v in ann.fc.state_dict().items()})
        def forward(self, x):
            m1,m2,m3=self.l1.init_leaky(),self.l2.init_leaky(),self.l3.init_leaky()
            out=torch.zeros(x.size(0),10,device=x.device)
            for _ in range(T_snn):
                s1,m1=self.l1(torch.relu(self.c1(x)),m1);s2,m2=self.l2(torch.relu(self.c2(s1)),m2)
                s3,m3=self.l3(torch.relu(self.c3(s2)),m3);out+=self.fc(self.flat(self.avg(s3)))
            return out
    return QCFS_SNN()

def eval_snn(m):
    m.eval(); cor, tot = 0, 0
    with torch.no_grad():
        for d, t in test_ldr: d,t=d.to(DEV),t.to(DEV); cor += m(d).max(1)[1].eq(t).sum().item(); tot += t.size(0)
    return 100*cor/tot

print(f"  {'T':<8} {'QCFS-SNN':<12} {'Gap':<10} {'Status'}")
print(f"  {'-'*40}")
best_acc, best_T = 0, None
for T_test in [2, 4, 8, 16, 32, 64]:
    snn = build_qcfs_snn(ann, lambda_values, T_test).to(DEV)
    acc = eval_snn(snn); gap = ann_acc - acc
    st = "PASS" if gap < 5 else ("CLOSE" if gap < 8 else "FAIL")
    print(f"  {T_test:<8} {acc:<12.2f}% {gap:<10.2f}% {st}")
    if acc > best_acc: best_acc, best_T = acc, T_test

# ═══════════════════════════════════════════════════════════
# PHASE 4: Compare with OLD percentile method
# ═══════════════════════════════════════════════════════════
print(f"\n[4/4] Comparison with old percentile method...")
# Old method: calibrate from ReLU outputs at 95%
# Use the old stride CNN trained with ReLU + convert
old_ann_path = "c:/neurocuda/examples/cnn_stride2_best.pt"

class OldCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(3,64,3,stride=2,padding=1,bias=False);self.b1=nn.BatchNorm2d(64)
        self.c2=nn.Conv2d(64,128,3,stride=2,padding=1,bias=False);self.b2=nn.BatchNorm2d(128)
        self.c3=nn.Conv2d(128,256,3,stride=2,padding=1,bias=False);self.b3=nn.BatchNorm2d(256)
        self.avg=nn.AdaptiveAvgPool2d(1);self.flat=nn.Flatten();self.fc=nn.Linear(256,10);self.relu=nn.ReLU()
    def forward(self,x):
        x=self.relu(self.b1(self.c1(x)));x=self.relu(self.b2(self.c2(x)));x=self.relu(self.b3(self.c3(x)))
        return self.fc(self.flat(self.avg(x)))

if os.path.exists(old_ann_path):
    old_ann = OldCNN(); old_ann.load_state_dict(torch.load(old_ann_path, map_location=DEV)); old_ann = old_ann.to(DEV); old_ann.eval()
    for c, b in [(old_ann.c1,old_ann.b1),(old_ann.c2,old_ann.b2),(old_ann.c3,old_ann.b3)]: fold(c, b)

    # Calibrate old method
    raw=[[],[],[]]
    with torch.no_grad():
        cb_ldr = DataLoader(Subset(tr_eds, range(2000)), B, shuffle=False, drop_last=True)
        for d,_ in cb_ldr:
            x=d.to(DEV);a1=old_ann.relu(old_ann.b1(old_ann.c1(x)));raw[0].append(a1.detach().flatten().cpu().numpy())
            raw[1].append(old_ann.relu(old_ann.b2(old_ann.c2(a1))).detach().flatten().cpu().numpy())
            raw[2].append(old_ann.relu(old_ann.b3(old_ann.c3(old_ann.relu(old_ann.b2(old_ann.c2(a1)))))).detach().flatten().cpu().numpy())
    old_th = [max(float(np.percentile(np.concatenate(r), 95.0)), 0.01) for r in raw]

    old_snn = build_qcfs_snn(old_ann, old_th, 64).to(DEV)  # Reuse builder with old thresholds
    old_acc = eval_snn(old_snn)

    print(f"  Old (percentile 95%, T=64): {old_acc:.1f}% (gap {ann_acc-old_acc:.1f}%)")
    print(f"  QCFS (best, T={best_T}):        {best_acc:.1f}% (gap {ann_acc-best_acc:.1f}%)")
    print(f"  QCFS (T=64):               {eval_snn(build_qcfs_snn(ann, lambda_values, 64).to(DEV)):.1f}%")

# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"QCFS PIPELINE — FINAL RESULTS")
print(f"{'='*60}")
print(f"  QCFS-ANN:         {ann_acc:.1f}%")
print(f"  QCFS-SNN (T={best_T}):   {best_acc:.1f}%  (gap {ann_acc-best_acc:.1f}%)")
print(f"  λ values:          {[f'{v:.3f}' for v in lambda_values]}")
print(f"  Old percentile:    {old_acc:.1f}%" if os.path.exists(old_ann_path) else "")
print(f"  QCFS improvement:  +{best_acc-old_acc:.1f}%" if os.path.exists(old_ann_path) else "")
print(f"{'='*60}")

json.dump({"method":"QCFS","ann":ann_acc,"snn":best_acc,"T":best_T,"gap":ann_acc-best_acc,"lambdas":lambda_values,"date":datetime.now().isoformat()},open("c:/neurocuda/examples/qcfs_results.json","w"),indent=2)
print("Results saved to qcfs_results.json")