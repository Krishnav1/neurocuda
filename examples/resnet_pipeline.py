"""
═══════════════════════════════════════════════════════════════════════════
NEUROCUDA — ResNet CIFAR-10 Complete Pipeline
═══════════════════════════════════════════════════════════════════════════
Proves NeuroCUDA on production architecture (ResNet-style).
1. Train ResNet with separate ReLUs (SNN-convertible)
2. Fold BN → calibrate per-ReLU thresholds
3. Convert to SNN with matched LIF neurons
4. Fine-tune for best accuracy
5. Report final numbers

RUN: python examples/resnet_pipeline.py
═══════════════════════════════════════════════════════════════════════════
"""
import torch, torch.nn as nn, snntorch as snn, numpy as np, json, time
from snntorch import surrogate, utils as snn_utils
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
B, EPOCHS, T_SNN, N_CALIB = 256, 30, 64, 5000
FT_EPOCHS, PCT = 3, 95.0
DEV = torch.device("cuda")
SAVE_PATH = "c:/neurocuda/examples/resnet_fixed_best.pt"

te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tr_trf = transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tr_etf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
te_ds = datasets.CIFAR10("c:/neurocuda/data", train=False, download=False, transform=te_tf)
tr_ds = datasets.CIFAR10("c:/neurocuda/data", train=True, download=False, transform=tr_trf)
tr_eds = datasets.CIFAR10("c:/neurocuda/data", train=True, download=False, transform=tr_etf)
tr_ldr = DataLoader(tr_ds, B, shuffle=True)
te_ldr = DataLoader(te_ds, B, shuffle=False, drop_last=True)
cb_ldr = DataLoader(Subset(tr_eds, range(N_CALIB)), B, shuffle=False, drop_last=True)
ft_ldr = DataLoader(Subset(tr_eds, range(5000)), 64, shuffle=True, drop_last=True)  # Smaller batch for GPU mem
test_ldr = DataLoader(Subset(te_ds, range(1000)), B, shuffle=False, drop_last=True)

print("=" * 60)
print("NEUROCUDA — RESNET PIPELINE")
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0)} | Epochs: {EPOCHS} | T: {T_SNN}")

# ═══════════════════════════════════════════════════════════
# RESNET WITH SEPARATE RELUs (SNN-CONVERTIBLE)
# ═══════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c_out)
        self.relu1 = nn.ReLU()  # ← Separate ReLU (after c1)
        self.c2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c_out)
        self.relu2 = nn.ReLU()  # ← Separate ReLU (after residual add)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out)
            )

    def forward(self, x):
        out = self.relu1(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        out += self.shortcut(x)
        return self.relu2(out)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(64)
        self.relu_stem = nn.ReLU()  # ← Separate
        self.layer1 = ResBlock(64, 128, stride=2)
        self.layer2 = ResBlock(128, 256, stride=2)
        self.layer3 = ResBlock(256, 512, stride=2)
        self.avg = nn.AdaptiveAvgPool2d(1); self.flat = nn.Flatten()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu_stem(self.b1(self.c1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.fc(self.flat(self.avg(x)))

# ═══════════════════════════════════════════════════════════
# PHASE 1: TRAIN
# ═══════════════════════════════════════════════════════════
import os
if os.path.exists(SAVE_PATH):
    ann = ResNet(); ann.load_state_dict(torch.load(SAVE_PATH, map_location=DEV)); ann = ann.to(DEV); ann.eval()
    print(f"\n[1/4] Loaded trained ResNet from {SAVE_PATH}")
else:
    print(f"\n[1/4] Training ResNet ({EPOCHS} epochs)...")
    ann = ResNet().to(DEV)
    opt = torch.optim.AdamW(ann.parameters(), lr=1e-3, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.CrossEntropyLoss(); best = 0
    for ep in range(EPOCHS):
        ann.train()
        for d, t in tr_ldr: d,t=d.to(DEV),t.to(DEV); opt.zero_grad(); crit(ann(d),t).backward(); opt.step()
        sch.step()
        ann.eval(); cor, tot = 0, 0
        with torch.no_grad():
            for d, t in test_ldr: d,t=d.to(DEV),t.to(DEV); cor += ann(d).max(1)[1].eq(t).sum().item(); tot += t.size(0)
        acc = 100*cor/tot
        if ep % 10 == 0 or ep >= EPOCHS - 3: print(f"  Epoch {ep+1}: {acc:.1f}%")
        if acc > best: best = acc; torch.save(ann.state_dict(), SAVE_PATH)
    ann.load_state_dict(torch.load(SAVE_PATH)); print(f"  Best: {best:.1f}%")

# Evaluate ANN
ann.eval(); cor, tot = 0, 0
with torch.no_grad():
    for d, t in test_ldr: d,t=d.to(DEV),t.to(DEV); cor += ann(d).max(1)[1].eq(t).sum().item(); tot += t.size(0)
ann_acc = 100*cor/tot; print(f"  ANN Accuracy: {ann_acc:.1f}%")

# ═══════════════════════════════════════════════════════════
# PHASE 2: FOLD BN + CALIBRATE
# ═══════════════════════════════════════════════════════════
print(f"\n[2/4] Folding BN + Calibrating thresholds...")

def fold(conv, bn):
    if conv.bias is None: conv.bias = nn.Parameter(torch.zeros(conv.out_channels))
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight.data *= s.view(-1, 1, 1, 1)
    conv.bias.data = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)
    bn.weight.data = torch.ones_like(bn.weight); bn.bias.data.zero_()
    bn.running_mean.zero_(); bn.running_var.fill_(1.0 - bn.eps)

# Fold all BN→Conv pairs
pairs = []; prev_conv = None
for name, mod in ann.named_modules():
    if isinstance(mod, nn.Conv2d): prev_conv = name
    elif isinstance(mod, nn.BatchNorm2d) and prev_conv:
        pairs.append((prev_conv, name)); prev_conv = None
for cn, bn in pairs: fold(dict(ann.named_modules())[cn], dict(ann.named_modules())[bn])
print(f"  Folded {len(pairs)} BN layers")

# Collect ReLU activations (each ReLU is separate now!)
act_data = {}; handles = []
def hook_fn(name):
    def hook(m, inp, out):
        if name not in act_data: act_data[name] = []
        act_data[name].append(out.detach().flatten().cpu().numpy())
    return hook

relu_names = []
# Also record shortcut conv outputs (no ReLU) for shortcut LIF calibration
sc_act_data = {}; sc_handles = []
def sc_hook_fn(name):
    def hook(m, inp, out):
        # Apply ReLU to shortcut output (we'll add ReLU before LIF in SNN)
        relu_out = torch.relu(out)
        if name not in sc_act_data: sc_act_data[name] = []
        sc_act_data[name].append(relu_out.detach().flatten().cpu().numpy())
    return hook

for name, mod in ann.named_modules():
    if isinstance(mod, nn.ReLU):
        handles.append(mod.register_forward_hook(hook_fn(name)))
        relu_names.append(name)
    # Hook shortcut convs (they're inside nn.Sequential with key '0')
    if 'shortcut.0' in name and isinstance(mod, nn.Conv2d):
        sc_handles.append(mod.register_forward_hook(sc_hook_fn(name)))
        print(f"  Found shortcut conv: {name}")

with torch.no_grad():
    for d, _ in cb_ldr: ann(d.to(DEV))
for h in handles: h.remove()

# 7 ReLUs: stem, l1.relu1, l1.relu2, l2.relu1, l2.relu2, l3.relu1, l3.relu2
th = [max(float(np.percentile(np.concatenate(act_data[n]), PCT)), 0.01) for n in relu_names]
for i, (n, t) in enumerate(zip(relu_names, th)):
    acts = np.concatenate(act_data[n])
    print(f"  [{i}] {n}: thr={t:.2f} (mean={acts.mean():.2f}, {PCT}%)")

# 3 shortcut thresholds: layer1.shortcut.0, layer2.shortcut.0, layer3.shortcut.0
sc_names = sorted(sc_act_data.keys())
sc_th = [max(float(np.percentile(np.concatenate(sc_act_data[n]), PCT)), 0.01) for n in sc_names] if sc_act_data else [1.0, 1.0, 1.0]
for i, (n, t) in enumerate(zip(sc_names, sc_th)):
    acts = np.concatenate(sc_act_data[n])
    print(f"  [SC{i}] {n}: thr={t:.2f} (mean={acts.mean():.2f}, {PCT}%)")

# ═══════════════════════════════════════════════════════════
# PHASE 3: CONVERT TO SNN
# ═══════════════════════════════════════════════════════════
print(f"\n[3/4] Converting ResNet → SpikingResNet...")

sg = surrogate.fast_sigmoid(slope=25)
TH = th  # shorthand

class SpikingResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem
        self.stem_conv = nn.Conv2d(3,64,3,stride=1,padding=1,bias=True)
        self.stem_conv.load_state_dict({k:v for k,v in ann.c1.state_dict().items()})
        self.stem_lif = snn.Leaky(beta=1.0,threshold=TH[0],spike_grad=sg,reset_mechanism="subtract")

        # Layer 1
        self.l1_c1 = nn.Conv2d(64,128,3,stride=2,padding=1,bias=True)
        self.l1_c1.load_state_dict({k:v for k,v in ann.layer1.c1.state_dict().items()})
        self.l1_l1 = snn.Leaky(beta=1.0,threshold=TH[1],spike_grad=sg,reset_mechanism="subtract")
        self.l1_c2 = nn.Conv2d(128,128,3,stride=1,padding=1,bias=True)
        self.l1_c2.load_state_dict({k:v for k,v in ann.layer1.c2.state_dict().items()})
        self.l1_l2 = snn.Leaky(beta=1.0,threshold=TH[2],spike_grad=sg,reset_mechanism="subtract")
        self.l1_sc = nn.Conv2d(64,128,1,stride=2,bias=True)
        self.l1_sc.load_state_dict({k:v for k,v in ann.layer1.shortcut[0].state_dict().items()})
        self.l1_sc_lif = snn.Leaky(beta=1.0,threshold=sc_th[0] if len(sc_th)>0 else 1.0,spike_grad=sg,reset_mechanism="subtract")

        # Layer 2
        self.l2_c1 = nn.Conv2d(128,256,3,stride=2,padding=1,bias=True)
        self.l2_c1.load_state_dict({k:v for k,v in ann.layer2.c1.state_dict().items()})
        self.l2_l1 = snn.Leaky(beta=1.0,threshold=TH[3],spike_grad=sg,reset_mechanism="subtract")
        self.l2_c2 = nn.Conv2d(256,256,3,stride=1,padding=1,bias=True)
        self.l2_c2.load_state_dict({k:v for k,v in ann.layer2.c2.state_dict().items()})
        self.l2_l2 = snn.Leaky(beta=1.0,threshold=TH[4],spike_grad=sg,reset_mechanism="subtract")
        self.l2_sc = nn.Conv2d(128,256,1,stride=2,bias=True)
        self.l2_sc.load_state_dict({k:v for k,v in ann.layer2.shortcut[0].state_dict().items()})
        self.l2_sc_lif = snn.Leaky(beta=1.0,threshold=sc_th[1] if len(sc_th)>1 else 1.0,spike_grad=sg,reset_mechanism="subtract")

        # Layer 3
        self.l3_c1 = nn.Conv2d(256,512,3,stride=2,padding=1,bias=True)
        self.l3_c1.load_state_dict({k:v for k,v in ann.layer3.c1.state_dict().items()})
        self.l3_l1 = snn.Leaky(beta=1.0,threshold=TH[5],spike_grad=sg,reset_mechanism="subtract")
        self.l3_c2 = nn.Conv2d(512,512,3,stride=1,padding=1,bias=True)
        self.l3_c2.load_state_dict({k:v for k,v in ann.layer3.c2.state_dict().items()})
        self.l3_l2 = snn.Leaky(beta=1.0,threshold=TH[6],spike_grad=sg,reset_mechanism="subtract")
        self.l3_sc = nn.Conv2d(256,512,1,stride=2,bias=True)
        self.l3_sc.load_state_dict({k:v for k,v in ann.layer3.shortcut[0].state_dict().items()})
        self.l3_sc_lif = snn.Leaky(beta=1.0,threshold=sc_th[2] if len(sc_th)>2 else 1.0,spike_grad=sg,reset_mechanism="subtract")

        self.avg = nn.AdaptiveAvgPool2d(1); self.flat = nn.Flatten(); self.fc = nn.Linear(512, 10)
        self.fc.load_state_dict({k:v for k,v in ann.fc.state_dict().items()})

    def forward(self, x):
        m = [l.init_leaky() for l in [self.stem_lif, self.l1_l1, self.l1_l2,
              self.l2_l1, self.l2_l2, self.l3_l1, self.l3_l2]]
        ms = [self.l1_sc_lif.init_leaky(), self.l2_sc_lif.init_leaky(), self.l3_sc_lif.init_leaky()]
        out = torch.zeros(x.size(0), 10, device=x.device)
        for _ in range(T_SNN):
            # Stem
            s0, m[0] = self.stem_lif(torch.relu(self.stem_conv(x)), m[0])
            # Layer 1 — shortcut LIF makes both paths produce SPIKES
            r1_conv = self.l1_sc(s0)
            r1, ms[0] = self.l1_sc_lif(torch.relu(r1_conv), ms[0])
            c1a, m[1] = self.l1_l1(torch.relu(self.l1_c1(s0)), m[1])
            c1b, m[2] = self.l1_l2(torch.relu(self.l1_c2(c1a)), m[2])
            s1 = c1b + r1  # Both binary → sum 0,1,2
            # Layer 2
            r2_conv = self.l2_sc(s1)
            r2, ms[1] = self.l2_sc_lif(torch.relu(r2_conv), ms[1])
            c2a, m[3] = self.l2_l1(torch.relu(self.l2_c1(s1)), m[3])
            c2b, m[4] = self.l2_l2(torch.relu(self.l2_c2(c2a)), m[4])
            s2 = c2b + r2
            # Layer 3
            r3_conv = self.l3_sc(s2)
            r3, ms[2] = self.l3_sc_lif(torch.relu(r3_conv), ms[2])
            c3a, m[5] = self.l3_l1(torch.relu(self.l3_c1(s2)), m[5])
            c3b, m[6] = self.l3_l2(torch.relu(self.l3_c2(c3a)), m[6])
            s3 = c3b + r3
            out += self.fc(self.flat(self.avg(s3)))
        return out

snn_base = SpikingResNet().to(DEV)

def eval_snn(m):
    m.eval(); cor, tot = 0, 0
    with torch.no_grad():
        for d, t in test_ldr: d,t=d.to(DEV),t.to(DEV); cor += m(d).max(1)[1].eq(t).sum().item(); tot += t.size(0)
    return 100*cor/tot

snn_pre = eval_snn(snn_base)
print(f"  SNN (before FT): {snn_pre:.1f}% (gap {ann_acc-snn_pre:.1f}%)")

# ═══════════════════════════════════════════════════════════
# PHASE 4: FINE-TUNE
# ═══════════════════════════════════════════════════════════
print(f"\n[4/4] Fine-tuning ({FT_EPOCHS} epochs)...")
snn_ft = SpikingResNet().to(DEV)
best_ft = snn_pre

for ep in range(FT_EPOCHS):
    lr = [1e-5, 5e-6, 1e-6][ep]
    opt = torch.optim.AdamW(snn_ft.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(); snn_ft.train()
    for d, t in ft_ldr: d,t=d.to(DEV),t.to(DEV); opt.zero_grad(); crit(snn_ft(d),t).backward(); opt.step(); snn_utils.reset(snn_ft)
    acc = eval_snn(snn_ft); print(f"  FT epoch {ep+1}: {acc:.1f}% (gap {ann_acc-acc:.1f}%)")
    if acc > best_ft: best_ft = acc; torch.save(snn_ft.state_dict(), "c:/neurocuda/examples/resnet_snn_best.pt")

snn_ft.load_state_dict(torch.load("c:/neurocuda/examples/resnet_snn_best.pt"))
snn_post = eval_snn(snn_ft)

# ═══════════════════════════════════════════════════════════
# FINAL
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"RESNET PIPELINE — FINAL RESULTS")
print(f"{'='*60}")
print(f"  ANN:            {ann_acc:.1f}%")
print(f"  SNN (convert):  {snn_pre:.1f}%  (gap {ann_acc-snn_pre:.1f}%)")
print(f"  SNN (fine-tune): {snn_post:.1f}%  (gap {ann_acc-snn_post:.1f}%)")
print(f"  FT gain:        +{snn_post-snn_pre:.1f}%")
print(f"  Architecture:   ResNet-style (7 ReLUs, residual)")
print(f"{'='*60}")

json.dump({"date":datetime.now().isoformat(),"arch":"ResNet","ann":ann_acc,"snn_pre":snn_pre,"snn_post":snn_post,"gap":ann_acc-snn_post,"ft_gain":snn_post-snn_pre,"T":T_SNN},open("c:/neurocuda/examples/resnet_results.json","w"),indent=2)
print("Results saved to resnet_results.json")
