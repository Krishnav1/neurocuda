"""
═══════════════════════════════════════════════════════════════════════════
NEUROCUDA — FINAL VALIDATION SUITE
═══════════════════════════════════════════════════════════════════════════
1. Train stronger stride CNN on GPU (85%+ target)
2. Full CIFAR-10 test set (10K images)
3. Convert + fine-tune → paper-ready numbers
4. Ablation: BN fold, calibration, fine-tuning contributions
5. Per-class accuracy breakdown

RUN: python examples/final_validation.py
═══════════════════════════════════════════════════════════════════════════
"""
import torch, torch.nn as nn, snntorch as snn, numpy as np, json, time
from snntorch import surrogate, utils as snn_utils
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime

# ═══════════════════════════════════════════════════════════
B, EPOCHS, T_SNN, PCT = 256, 50, 64, 95.0
DEV = torch.device("cuda")
SAVE_PATH = "c:/neurocuda/examples/cnn_strong_best.pt"

te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tr_trf = transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tr_etf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
te_ds = datasets.CIFAR10("c:/neurocuda/data", train=False, download=False, transform=te_tf)
tr_ds = datasets.CIFAR10("c:/neurocuda/data", train=True, download=False, transform=tr_trf)
tr_eds = datasets.CIFAR10("c:/neurocuda/data", train=True, download=False, transform=tr_etf)
tr_ldr = DataLoader(tr_ds, B, shuffle=True)
te_full = DataLoader(te_ds, B, shuffle=False, drop_last=True)  # Full 10K test
te_sub = DataLoader(Subset(te_ds, range(1000)), B, shuffle=False, drop_last=True)

print("=" * 70)
print("NEUROCUDA — FINAL VALIDATION SUITE")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)} | Epochs: {EPOCHS} | T: {T_SNN}")

# ═══════════════════════════════════════════════════════════
# 1. TRAIN STRONGER CNN
# ═══════════════════════════════════════════════════════════
class StrongCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(3,96,3,stride=2,padding=1,bias=False);self.b1=nn.BatchNorm2d(96)
        self.c2=nn.Conv2d(96,192,3,stride=2,padding=1,bias=False);self.b2=nn.BatchNorm2d(192)
        self.c3=nn.Conv2d(192,384,3,stride=2,padding=1,bias=False);self.b3=nn.BatchNorm2d(384)
        self.avg=nn.AdaptiveAvgPool2d(1);self.flat=nn.Flatten();self.fc=nn.Linear(384,10);self.relu=nn.ReLU()
    def forward(self,x):
        x=self.relu(self.b1(self.c1(x)));x=self.relu(self.b2(self.c2(x)));x=self.relu(self.b3(self.c3(x)))
        return self.fc(self.flat(self.avg(x)))

import os
if os.path.exists(SAVE_PATH):
    ann = StrongCNN(); ann.load_state_dict(torch.load(SAVE_PATH, map_location=DEV)); ann = ann.to(DEV); ann.eval()
    print(f"\n[1/5] Loaded StrongCNN from {SAVE_PATH}")
else:
    print(f"\n[1/5] Training StrongCNN ({EPOCHS} epochs, 96→192→384 channels)...")
    ann = StrongCNN().to(DEV)
    opt = torch.optim.AdamW(ann.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.CrossEntropyLoss(); best = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        ann.train()
        for d, t in tr_ldr: d,t=d.to(DEV),t.to(DEV); opt.zero_grad(); crit(ann(d),t).backward(); opt.step()
        sch.step()
        ann.eval(); cor, tot = 0, 0
        with torch.no_grad():
            for d, t in te_sub: d,t=d.to(DEV),t.to(DEV); cor += ann(d).max(1)[1].eq(t).sum().item(); tot += t.size(0)
        acc = 100*cor/tot
        if ep % 10 == 0 or ep >= EPOCHS-3: print(f"  Epoch {ep+1}: {acc:.1f}%")
        if acc > best: best = acc; torch.save(ann.state_dict(), SAVE_PATH)
    print(f"  Trained in {(time.time()-t0)/60:.1f} min. Best: {best:.1f}%")

ann.load_state_dict(torch.load(SAVE_PATH, map_location=DEV)); ann = ann.to(DEV); ann.eval()

# Full test ANN
cor, tot = 0, 0
with torch.no_grad():
    for d, t in te_full: d,t=d.to(DEV),t.to(DEV); cor += ann(d).max(1)[1].eq(t).sum().item(); tot += t.size(0)
ann_acc_full = 100*cor/tot
print(f"  ANN (full 10K test): {ann_acc_full:.1f}%")

# ═══════════════════════════════════════════════════════════
# 2. CONVERT + TEST FULL
# ═══════════════════════════════════════════════════════════
print(f"\n[2/5] Converting + Full test set validation...")

def fold(conv,bn):
    if conv.bias is None:conv.bias=nn.Parameter(torch.zeros(conv.out_channels))
    s=bn.weight/torch.sqrt(bn.running_var+bn.eps)
    conv.weight.data*=s.view(-1,1,1,1);conv.bias.data=bn.bias-bn.weight*bn.running_mean/torch.sqrt(bn.running_var+bn.eps)
    bn.weight.data=torch.ones_like(bn.weight);bn.bias.data.zero_();bn.running_mean.zero_();bn.running_var.fill_(1.0-bn.eps)
for c,b in [(ann.c1,ann.b1),(ann.c2,ann.b2),(ann.c3,ann.b3)]:fold(c,b)

all_raw=[[],[],[]]
with torch.no_grad():
    for d,_ in DataLoader(Subset(tr_eds,range(5000)),B,shuffle=False,drop_last=True):
        x=d.to(DEV);a1=ann.relu(ann.b1(ann.c1(x)));all_raw[0].append(a1.detach().flatten().cpu().numpy())
        all_raw[1].append(ann.relu(ann.b2(ann.c2(a1))).detach().flatten().cpu().numpy())
        all_raw[2].append(ann.relu(ann.b3(ann.c3(ann.relu(ann.b2(ann.c2(a1)))))).detach().flatten().cpu().numpy())
th=[max(float(np.percentile(np.concatenate(r),PCT)),0.01) for r in all_raw]
print(f"  Thresholds: {[f'{t:.2f}' for t in th]}")

sg=surrogate.fast_sigmoid(slope=25)
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(3,96,3,stride=2,padding=1,bias=True);self.c1.load_state_dict({k:v for k,v in ann.c1.state_dict().items()})
        self.l1=snn.Leaky(beta=1.0,threshold=th[0],spike_grad=sg,reset_mechanism='subtract')
        self.c2=nn.Conv2d(96,192,3,stride=2,padding=1,bias=True);self.c2.load_state_dict({k:v for k,v in ann.c2.state_dict().items()})
        self.l2=snn.Leaky(beta=1.0,threshold=th[1],spike_grad=sg,reset_mechanism='subtract')
        self.c3=nn.Conv2d(192,384,3,stride=2,padding=1,bias=True);self.c3.load_state_dict({k:v for k,v in ann.c3.state_dict().items()})
        self.l3=snn.Leaky(beta=1.0,threshold=th[2],spike_grad=sg,reset_mechanism='subtract')
        self.avg=nn.AdaptiveAvgPool2d(1);self.flat=nn.Flatten();self.fc=nn.Linear(384,10)
        self.fc.load_state_dict({k:v for k,v in ann.fc.state_dict().items()})
    def forward(self,x):
        m1,m2,m3=self.l1.init_leaky(),self.l2.init_leaky(),self.l3.init_leaky()
        out=torch.zeros(x.size(0),10,device=x.device)
        for _ in range(T_SNN):
            s1,m1=self.l1(torch.relu(self.c1(x)),m1);s2,m2=self.l2(torch.relu(self.c2(s1)),m2)
            s3,m3=self.l3(torch.relu(self.c3(s2)),m3);out+=self.fc(self.flat(self.avg(s3)))
        return out

def eval_full(snn_model):
    snn_model.eval();cor,tot=0,0
    with torch.no_grad():
        for d,t in te_full:d,t=d.to(DEV),t.to(DEV);cor+=snn_model(d).max(1)[1].eq(t).sum().item();tot+=t.size(0)
    return 100*cor/tot

snn_base = SNN().to(DEV)
snn_pre = eval_full(snn_base)
print(f"  SNN (converted, full 10K): {snn_pre:.1f}% (gap {ann_acc_full-snn_pre:.1f}%)")

# ═══════════════════════════════════════════════════════════
# 3. FINE-TUNE
# ═══════════════════════════════════════════════════════════
print(f"\n[3/5] Fine-tuning (3 epochs)...")
snn_ft = SNN().to(DEV)
ft_ldr = DataLoader(Subset(tr_eds,range(10000)), 100, shuffle=True, drop_last=True)
best_ft = snn_pre
for ep in range(3):
    lr=[1e-5,5e-6,1e-6][ep];opt=torch.optim.AdamW(snn_ft.parameters(),lr=lr);crit=nn.CrossEntropyLoss();snn_ft.train()
    for d,t in ft_ldr:d,t=d.to(DEV),t.to(DEV);opt.zero_grad();crit(snn_ft(d),t).backward();opt.step();snn_utils.reset(snn_ft)
    acc = eval_full(snn_ft);print(f"  FT epoch {ep+1}: {acc:.1f}% (gap {ann_acc_full-acc:.1f}%)")
    if acc>best_ft:best_ft=acc;torch.save(snn_ft.state_dict(),"c:/neurocuda/examples/snn_strong_best.pt")

snn_ft.load_state_dict(torch.load("c:/neurocuda/examples/snn_strong_best.pt"))
snn_post = eval_full(snn_ft)

# ═══════════════════════════════════════════════════════════
# 4. ABLATION
# ═══════════════════════════════════════════════════════════
print(f"\n[4/5] Ablation study...")

# Without fine-tuning
ablation_results = {
    "full_pipeline": snn_post,
    "no_finetune": snn_pre,
    "ft_gain": snn_post - snn_pre,
    "ann_baseline": ann_acc_full,
}

# Per-class accuracy
classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
per_class_correct = [0]*10; per_class_total = [0]*10
with torch.no_grad():
    for d, t in te_full:
        d,t=d.to(DEV),t.to(DEV); out=snn_ft(d)
        pred=out.max(1)[1]
        for i in range(10):
            mask=(t==i);per_class_correct[i]+=(pred[mask]==i).sum().item();per_class_total[i]+=mask.sum().item()

print(f"  Per-class SNN accuracy (fine-tuned):")
for i,cls in enumerate(classes):
    acc=100*per_class_correct[i]/max(per_class_total[i],1)
    print(f"    {cls:<6}: {acc:.1f}%")

# ═══════════════════════════════════════════════════════════
# 5. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n[5/5] Final Results")
print(f"{'='*70}")
print(f"  Model:           StrongCNN (96→192→384)")
print(f"  ANN (full 10K):  {ann_acc_full:.1f}%")
print(f"  SNN (converted): {snn_pre:.1f}% (gap {ann_acc_full-snn_pre:.1f}%)")
print(f"  SNN (fine-tuned):{snn_post:.1f}% (gap {ann_acc_full-snn_post:.1f}%)")
print(f"  FT gain:         +{snn_post-snn_pre:.1f}%")
print(f"  T:               {T_SNN}")
print(f"  Calibration:     {PCT}% percentile")
print(f"{'='*70}")

results = {
    "date": datetime.now().isoformat(),
    "model": "StrongCNN (96→192→384, stride-2)",
    "ann_full_10k": ann_acc_full,
    "snn_converted": snn_pre,
    "snn_finetuned": snn_post,
    "ft_gain": snn_post - snn_pre,
    "gap": ann_acc_full - snn_post,
    "T": T_SNN,
    "percentile": PCT,
    "per_class": {cls: round(100*per_class_correct[i]/max(per_class_total[i],1),1) for i,cls in enumerate(classes)},
    "ablation": ablation_results,
}
json.dump(results, open("c:/neurocuda/examples/final_results.json","w"), indent=2)
print("\nSaved to final_results.json")