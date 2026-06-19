"""
QCFS From-Scratch Training (NOT fine-tuning)
=============================================
Trains stride CNN with QCFS activations from scratch.
Lambdas learn naturally during training.

Following Bu et al., ICLR 2022:
  - Small initial L (4), increase to 16
  - Train with QCFS throughout
  - 100 epochs total

Target: 80%+ ANN, then convert to SNN at T=4-8 with <5% gap.
"""
import torch, torch.nn as nn, snntorch as snn, numpy as np, json
from snntorch import surrogate
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurocuda.qcfs import QCFSActivation

B, EPOCHS, L_INIT, L_FINAL = 256, 100, 4, 16
DEV = torch.device("cuda")

te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tr_tf = transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
te_ds = datasets.CIFAR10("c:/neurocuda/data", train=False, download=False, transform=te_tf)
tr_ds = datasets.CIFAR10("c:/neurocuda/data", train=True, download=False, transform=tr_tf)
tr_ldr = DataLoader(tr_ds, B, shuffle=True)
test_ldr = DataLoader(Subset(te_ds, range(1000)), B, shuffle=False, drop_last=True)

class QCNN(nn.Module):
    def __init__(self, L=16):
        super().__init__()
        self.c1=nn.Conv2d(3,64,3,stride=2,padding=1,bias=False);self.b1=nn.BatchNorm2d(64);self.q1=QCFSActivation(1.0,L)
        self.c2=nn.Conv2d(64,128,3,stride=2,padding=1,bias=False);self.b2=nn.BatchNorm2d(128);self.q2=QCFSActivation(1.0,L)
        self.c3=nn.Conv2d(128,256,3,stride=2,padding=1,bias=False);self.b3=nn.BatchNorm2d(256);self.q3=QCFSActivation(1.0,L)
        self.avg=nn.AdaptiveAvgPool2d(1);self.flat=nn.Flatten();self.fc=nn.Linear(256,10)
    def forward(self,x):
        x=self.q1(self.b1(self.c1(x)));x=self.q2(self.b2(self.c2(x)));x=self.q3(self.b3(self.c3(x)))
        return self.fc(self.flat(self.avg(x)))

print(f"QCFS FROM SCRATCH — {EPOCHS} epochs, L={L_INIT}→{L_FINAL}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

ann = QCNN(L=L_INIT).to(DEV)
opt = torch.optim.AdamW(ann.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
crit = nn.CrossEntropyLoss(); best = 0

# Gradual L increase schedule
L_INCREASE_EPOCH = EPOCHS // 3  # At 1/3 point, increase L

for ep in range(EPOCHS):
    # Increase L at scheduled epoch
    if ep == L_INCREASE_EPOCH:
        for name, mod in ann.named_modules():
            if isinstance(mod, QCFSActivation):
                mod.L = L_FINAL
        print(f"  Epoch {ep+1}: Increased L from {L_INIT} to {L_FINAL}")

    ann.train()
    for d, t in tr_ldr: d,t=d.to(DEV),t.to(DEV); opt.zero_grad(); crit(ann(d),t).backward(); opt.step()
    sch.step()

    ann.eval(); cor, tot = 0, 0
    with torch.no_grad():
        for d, t in test_ldr: d,t=d.to(DEV),t.to(DEV); cor += ann(d).max(1)[1].eq(t).sum().item(); tot += t.size(0)
    acc = 100*cor/tot
    if ep % 20 == 0 or ep >= EPOCHS-5 or ep == L_INCREASE_EPOCH:
        lams = []
        for m in ann.modules():
            if isinstance(m, QCFSActivation): lams.append(f"{m.get_threshold().item():.3f}")
        print(f"  Epoch {ep+1}: {acc:.1f}%  λ=[{', '.join(lams)}]")
    if acc > best: best = acc

# Save
torch.save({"state_dict": ann.state_dict(), "lambdas": lams, "accuracy": best},
           "c:/neurocuda/examples/qcfs_scratch_best.pt")
print(f"Best: {best:.1f}% saved")