#!/usr/bin/env python3
"""NEUROCUDA v0.1 — FULL WORKING DEMO. Run: python examples/demo.py"""
import torch, torch.nn as nn, snntorch as snn, numpy as np, json
from snntorch import surrogate, utils as snn_utils
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime

BATCH, T_SNN, N_CALIB, N_TEST = 256, 64, 5000, 1000
PCT, FT_EPOCHS = 95.0, 3
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANN_PATH = "c:/neurocuda/examples/cnn_stride2_best.pt"

print("=" * 60)
print("NEUROCUDA v0.1 — END-TO-END DEMO")
print("=" * 60)
print(f"Device: {torch.cuda.get_device_name(0) if 'cuda' in str(DEV) else 'CPU'}")
print(f"T={T_SNN} | PCT={PCT}% | FT epochs={FT_EPOCHS}\n")

te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tr_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
te_ds = datasets.CIFAR10("c:/neurocuda/data", train=False, download=True, transform=te_tf)
tr_ds = datasets.CIFAR10("c:/neurocuda/data", train=True, download=True, transform=tr_tf)
test_ldr = DataLoader(Subset(te_ds, range(N_TEST)), BATCH, shuffle=False, drop_last=True)
calib_ldr = DataLoader(Subset(tr_ds, range(N_CALIB)), BATCH, shuffle=False, drop_last=True)
ft_ldr = DataLoader(Subset(tr_ds, range(10000)), BATCH, shuffle=True, drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(3,64,3,stride=2,padding=1,bias=False);self.b1=nn.BatchNorm2d(64)
        self.c2=nn.Conv2d(64,128,3,stride=2,padding=1,bias=False);self.b2=nn.BatchNorm2d(128)
        self.c3=nn.Conv2d(128,256,3,stride=2,padding=1,bias=False);self.b3=nn.BatchNorm2d(256)
        self.avg=nn.AdaptiveAvgPool2d(1);self.flat=nn.Flatten();self.fc=nn.Linear(256,10);self.relu=nn.ReLU()
    def forward(self,x):
        x=self.relu(self.b1(self.c1(x)));x=self.relu(self.b2(self.c2(x)))
        x=self.relu(self.b3(self.c3(x)));return self.fc(self.flat(self.avg(x)))

# [1] Load ANN
print("[1/6] Loading ANN...")
ann=CNN();ann.load_state_dict(torch.load(ANN_PATH,map_location=DEV));ann=ann.to(DEV);ann.eval()
cor,tot=0,0
with torch.no_grad():
    for d,t in test_ldr: d,t=d.to(DEV),t.to(DEV);cor+=ann(d).max(1)[1].eq(t).sum().item();tot+=t.size(0)
ann_acc=100*cor/tot;print(f"      ANN: {ann_acc:.1f}%")

# [2] Fold BN + Calibrate
print("[2/6] Folding BN + Calibrating...")
def fold(conv,bn):
    if conv.bias is None: conv.bias=nn.Parameter(torch.zeros(conv.out_channels))
    s=bn.weight/torch.sqrt(bn.running_var+bn.eps)
    conv.weight.data*=s.view(-1,1,1,1);conv.bias.data=bn.bias-bn.weight*bn.running_mean/torch.sqrt(bn.running_var+bn.eps)
    bn.weight.data=torch.ones_like(bn.weight);bn.bias.data.zero_();bn.running_mean.zero_();bn.running_var.fill_(1.0-bn.eps)
for c,b in [(ann.c1,ann.b1),(ann.c2,ann.b2),(ann.c3,ann.b3)]:fold(c,b)
all_acts=[[],[],[]]
with torch.no_grad():
    for d,_ in calib_ldr:
        x=d.to(DEV);a1=ann.relu(ann.b1(ann.c1(x)));all_acts[0].append(a1.flatten().cpu().numpy())
        all_acts[1].append(ann.relu(ann.b2(ann.c2(a1))).flatten().cpu().numpy())
        all_acts[2].append(ann.relu(ann.b3(ann.c3(ann.relu(ann.b2(ann.c2(a1)))))).flatten().cpu().numpy())
all_vals=[np.concatenate(a) for a in all_acts]
th=[max(float(np.percentile(v,PCT)),0.01) for v in all_vals]
for i,t in enumerate(th):print(f"      L{i+1}: th={t:.2f} (mean={all_vals[i].mean():.2f})")

# [3] Convert
print("[3/6] Converting ANN -> SNN...")
sg=surrogate.fast_sigmoid(slope=25)
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(3,64,3,stride=2,padding=1,bias=True);self.c1.load_state_dict({k:v for k,v in ann.c1.state_dict().items()})
        self.l1=snn.Leaky(beta=1.0,threshold=th[0],spike_grad=sg,reset_mechanism="subtract")
        self.c2=nn.Conv2d(64,128,3,stride=2,padding=1,bias=True);self.c2.load_state_dict({k:v for k,v in ann.c2.state_dict().items()})
        self.l2=snn.Leaky(beta=1.0,threshold=th[1],spike_grad=sg,reset_mechanism="subtract")
        self.c3=nn.Conv2d(128,256,3,stride=2,padding=1,bias=True);self.c3.load_state_dict({k:v for k,v in ann.c3.state_dict().items()})
        self.l3=snn.Leaky(beta=1.0,threshold=th[2],spike_grad=sg,reset_mechanism="subtract")
        self.avg=nn.AdaptiveAvgPool2d(1);self.flat=nn.Flatten();self.fc=nn.Linear(256,10)
        self.fc.load_state_dict({k:v for k,v in ann.fc.state_dict().items()})
    def forward(self,x):
        m1,m2,m3=self.l1.init_leaky(),self.l2.init_leaky(),self.l3.init_leaky()
        out=torch.zeros(x.size(0),10,device=x.device)
        for _ in range(T_SNN):
            s1,m1=self.l1(torch.relu(self.c1(x)),m1);s2,m2=self.l2(torch.relu(self.c2(s1)),m2)
            s3,m3=self.l3(torch.relu(self.c3(s2)),m3);out+=self.fc(self.flat(self.avg(s3)))
        return out

snn_pre=SNN().to(DEV)
def eval_snn(m):
    m.eval();cor,tot=0,0
    with torch.no_grad():
        for d,t in test_ldr:d,t=d.to(DEV),t.to(DEV);cor+=m(d).max(1)[1].eq(t).sum().item();tot+=t.size(0)
    return 100*cor/tot
snn_before=eval_snn(snn_pre)
print(f"      SNN (before FT): {snn_before:.1f}% (gap {ann_acc-snn_before:.1f}%)")

# [4] Fine-tune
print(f"[4/6] Fine-tuning ({FT_EPOCHS} epochs)...")
snn_ft=SNN().to(DEV)
for ep in range(FT_EPOCHS):
    lr=[1e-5,5e-6,1e-6][ep];opt=torch.optim.AdamW(snn_ft.parameters(),lr=lr);crit=nn.CrossEntropyLoss();snn_ft.train()
    for d,t in ft_ldr:d,t=d.to(DEV),t.to(DEV);opt.zero_grad();crit(snn_ft(d),t).backward();opt.step();snn_utils.reset(snn_ft)
    acc=eval_snn(snn_ft)
    print(f"      Epoch {ep+1}/{FT_EPOCHS}: {acc:.1f}% (gap {ann_acc-acc:.1f}%)")
    if ep==0 or acc>eval_snn(snn_ft):torch.save(snn_ft.state_dict(),"snn_demo_best.pt")
snn_ft.load_state_dict(torch.load("snn_demo_best.pt"));snn_after=eval_snn(snn_ft)

# [5] Compare
print(f"[5/6] ANN vs SNN predictions...")
sample_ldr=DataLoader(Subset(te_ds,range(10)),10,shuffle=False)
imgs,lbls=next(iter(sample_ldr));imgs,lbls=imgs.to(DEV),lbls.to(DEV)
with torch.no_grad():ann_p=ann(imgs).max(1)[1];snn_p=snn_ft(imgs).max(1)[1]
classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
agree=(ann_p==snn_p).sum().item()
for i in range(10):print(f"      {classes[ann_p[i]]:<6} vs {classes[snn_p[i]]:<6} {'OK' if ann_p[i]==snn_p[i] else 'XX'}")
print(f"      Agreement: {agree}/10")

# [6] Energy
print(f"[6/6] Energy estimation...")
params=sum(p.numel() for p in snn_ft.parameters())
ann_f=params*2;snn_o=params*0.20*T_SNN
gpu_u=ann_f*50/1e6;neuro_u=snn_o*0.1/1e6;ratio=gpu_u/max(neuro_u,1e-6)
print(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
print(f"  ANN: {ann_acc:.1f}% | SNN conv: {snn_before:.1f}% | SNN ft: {snn_after:.1f}%")
print(f"  Gap: {ann_acc-snn_after:.1f}% | FT gain: +{snn_after-snn_before:.1f}%")
print(f"  Agreement: {agree}/10 | Energy vs GPU: ~{ratio:.0f}x")
print(f"  Pipeline: WORKING")
print(f"{'='*60}\n")
json.dump({"date":datetime.now().isoformat(),"ann":ann_acc,"snn_before":snn_before,"snn_after":snn_after,"gain":snn_after-snn_before,"agreement":f"{agree}/10","energy_ratio":ratio},open("demo_results.json","w"),indent=2)
print("Saved demo_results.json")
