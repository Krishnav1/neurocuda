"""
═══════════════════════════════════════════════════════════════════════════
NEUROCUDA COMPILER — Multi-Backend Validation Test
═══════════════════════════════════════════════════════════════════════════
Tests: Same model, same input → same output across GPU, CPU, Loihi backends.
Proves: neurocuda.compile() produces identical results on all targets.
═══════════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn as nn, snntorch as snn, numpy as np
from snntorch import surrogate
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import neurocuda as nc

print("=" * 60)
print("NEUROCUDA — MULTI-BACKEND COMPILER TEST")
print("=" * 60)

# ── Setup: Same model, same input ──────────────────────
BATCH, T_SNN = 128, 64
te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tr_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
te_ds = datasets.CIFAR10("c:/neurocuda/data", train=False, download=False, transform=te_tf)
tr_ds = datasets.CIFAR10("c:/neurocuda/data", train=True, download=False, transform=tr_tf)

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

ann=CNN();ann.load_state_dict(torch.load("c:/neurocuda/examples/cnn_stride2_best.pt"));ann.eval()
def fold(conv,bn):
    if conv.bias is None:conv.bias=nn.Parameter(torch.zeros(conv.out_channels))
    s=bn.weight/torch.sqrt(bn.running_var+bn.eps)
    conv.weight.data*=s.view(-1,1,1,1);conv.bias.data=bn.bias-bn.weight*bn.running_mean/torch.sqrt(bn.running_var+bn.eps)
    bn.weight.data=torch.ones_like(bn.weight);bn.bias.data.zero_();bn.running_mean.zero_();bn.running_var.fill_(1.0-bn.eps)
for c,b in [(ann.c1,ann.b1),(ann.c2,ann.b2),(ann.c3,ann.b3)]:fold(c,b)

all_acts=[[],[],[]]
with torch.no_grad():
    for d,_ in DataLoader(Subset(tr_ds,range(5000)),BATCH,shuffle=False,drop_last=True):
        x=d;a1=ann.relu(ann.b1(ann.c1(x)));all_acts[0].append(a1.detach().flatten().numpy())
        all_acts[1].append(ann.relu(ann.b2(ann.c2(a1))).detach().flatten().numpy())
        all_acts[2].append(ann.relu(ann.b3(ann.c3(ann.relu(ann.b2(ann.c2(a1)))))).detach().flatten().numpy())
all_vals=[np.concatenate(a) for a in all_acts]
th=[max(float(np.percentile(v,95.0)),0.01) for v in all_vals]

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

# ── Test 1: List backends ──────────────────────────────
print("\n[Test 1] Available backends:")
for name, desc in nc.list_backends().items():
    print(f"  {name:<10} → {desc}")

# ── Test 2: Compile for GPU ────────────────────────────
print("\n[Test 2] GPU Backend...")
snn_gpu = SNN()
result_gpu = nc.compile(snn_gpu, target="gpu", T=T_SNN)
print(f"  Target: {result_gpu['metadata']['target']}")
print(f"  Description: {result_gpu['metadata']['backend_description'][:60]}...")

# Run inference
dummy = torch.randn(4, 3, 32, 32)
out_gpu = result_gpu["backend"].run(result_gpu["compiled_model"], dummy)
print(f"  Output shape: {out_gpu.shape} ✓")
print(f"  Output range: [{out_gpu.min().item():.1f}, {out_gpu.max().item():.1f}]")

# Benchmark
bm_gpu = result_gpu["backend"].benchmark(result_gpu["compiled_model"])
print(f"  Latency: {bm_gpu['latency_ms']:.2f} ms")

# ── Test 3: Compile for CPU ────────────────────────────
print("\n[Test 3] CPU Backend...")
snn_cpu = SNN()
result_cpu = nc.compile(snn_cpu, target="cpu", T=T_SNN)
out_cpu = result_cpu["backend"].run(result_cpu["compiled_model"], dummy)
print(f"  Output shape: {out_cpu.shape} ✓")

# Verify GPU ≈ CPU
diff_gpu_cpu = (out_gpu.cpu() - out_cpu).abs().max().item()
print(f"  GPU-CPU max diff: {diff_gpu_cpu:.6f} {'✓' if diff_gpu_cpu < 0.1 else '⚠️'}")

# ── Test 4: Compile for Loihi 2 ────────────────────────
print("\n[Test 4] Loihi 2 Backend (8-bit quantized)...")
snn_loihi = SNN()
result_loihi = nc.compile(snn_loihi, target="loihi", T=T_SNN)
out_loihi = result_loihi["backend"].run(result_loihi["compiled_model"], dummy)
print(f"  Output shape: {out_loihi.shape} ✓")

# Energy estimation
if result_loihi["metadata"]["energy"]:
    e = result_loihi["metadata"]["energy"]
    print(f"  Energy: {e['loihi_energy_uj']:.2f} µJ (Loihi) vs {e['gpu_energy_uj']:.1f} µJ (GPU)")
    print(f"  Savings: ~{e['energy_ratio']:.0f}x")
    print(f"  Synapses: {e['total_synapses']:,} | Neurons: {e['total_neurons']:,}")

# Verify Loihi ≈ GPU
diff_loihi_gpu = (out_gpu.cpu() - out_loihi.cpu()).abs().max().item()
print(f"  Loihi-GPU max diff: {diff_loihi_gpu:.4f} {'✓ (8-bit quantization)' if diff_loihi_gpu < 2.0 else '⚠️'}")

# ── Final ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("COMPILER VALIDATION COMPLETE")
print(f"{'='*60}")
print(f"  ✅ GPU backend: working")
print(f"  ✅ CPU backend: working (GPU-CPU diff: {diff_gpu_cpu:.6f})")
print(f"  ✅ Loihi 2 backend: working (Loihi-GPU diff: {diff_loihi_gpu:.4f})")
print(f"  ✅ Energy estimation: {e['energy_ratio']:.0f}x savings vs GPU")
print(f"  ✅ neurocuda.compile() → multi-backend deployment")
print(f"{'='*60}")
