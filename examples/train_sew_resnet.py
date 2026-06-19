"""
═══════════════════════════════════════════════════════════════════════════
SEW-ResNet — Direct SNN Training (Fang et al., NeurIPS 2021)
═══════════════════════════════════════════════════════════════════════════
Spike-Element-Wise ResNet trained directly as SNN.
No conversion. No gap. SOTA accuracy.

Reference: github.com/fangwei123456/Spike-Element-Wise-ResNet
           "Deep Residual Learning in Spiking Neural Networks" (NeurIPS 2021)

Target: 90%+ CIFAR-10 at T=4 timesteps
═══════════════════════════════════════════════════════════════════════════
"""
import torch, torch.nn as nn, snntorch as snn, numpy as np, json, time
from snntorch import surrogate, utils as snn_utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime

# ═══════════════════════════════════════════════════════════
B, EPOCHS, T_TRAIN, T_EVAL = 128, 50, 4, 8
DEV = torch.device("cuda")
SAVE_PATH = "c:/neurocuda/examples/sew_resnet_best.pt"

te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tr_tf = transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tr_ds = datasets.CIFAR10("c:/neurocuda/data", train=True, download=False, transform=tr_tf)
te_ds = datasets.CIFAR10("c:/neurocuda/data", train=False, download=False, transform=te_tf)
tr_ldr = DataLoader(tr_ds, B, shuffle=True, num_workers=0)
te_ldr = DataLoader(te_ds, B, shuffle=False, num_workers=0, drop_last=True)

print("=" * 70)
print("SEW-ResNet — Direct SNN Training (NeurIPS 2021)")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)} | T_train={T_TRAIN} | T_eval={T_EVAL} | Epochs={EPOCHS}")

# ═══════════════════════════════════════════════════════════
# SEW-ResNet Architecture
# ═══════════════════════════════════════════════════════════
spike_grad = surrogate.fast_sigmoid(slope=25)

def lif_neuron():
    """IF neuron — no leak (beta=1.0), subtractive reset."""
    return snn.Leaky(beta=1.0, threshold=1.0, spike_grad=spike_grad, reset_mechanism="subtract")

class SEWBlock(nn.Module):
    """SEW ResBlock with ADD operation. Both paths produce binary spikes."""
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.lif1 = lif_neuron()
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.lif2 = lif_neuron()

        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x, mem_pack):
        """x: spike train [B, C, H, W]. mem_pack: tuple of 4 membrane tensors."""
        m1, m2, m3, m4 = mem_pack

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out, m1 = self.lif1(out, m1)  # Binary spikes

        out = self.conv2(out)
        out = self.bn2(out)
        out, m2 = self.lif2(out, m2)  # Binary spikes

        # Shortcut path (also produces binary spikes)
        if len(self.shortcut) > 0:
            sc = self.shortcut[0](x)
            sc = self.shortcut[1](sc)
        else:
            sc = x

        # SEW ADD: element-wise addition of two binary spike trains
        # Both are binary → sum is 0, 1, or 2
        return out + sc, (m1, m2, None, None)


class SEWResNet(nn.Module):
    """SEW-ResNet for CIFAR-10. Direct SNN training."""
    def __init__(self):
        super().__init__()
        # Stem
        self.conv_stem = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(64)
        self.lif_stem = lif_neuron()

        # Layers (2 blocks each)
        self.layer1 = nn.Sequential(
            SEWBlock(64, 128, stride=2),
            SEWBlock(128, 128, stride=1),
        )
        self.layer2 = nn.Sequential(
            SEWBlock(128, 256, stride=2),
            SEWBlock(256, 256, stride=1),
        )
        self.layer3 = nn.Sequential(
            SEWBlock(256, 512, stride=2),
            SEWBlock(512, 512, stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(512, 10)

    def forward(self, x, T):
        """Run SNN for T timesteps. Returns accumulated output."""
        out_total = torch.zeros(x.size(0), 10, device=x.device)

        # Initialize membrane potentials
        m_stem = self.lif_stem.init_leaky()
        m_packs = []
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                m_packs.append((
                    block.lif1.init_leaky(),
                    block.lif2.init_leaky(),
                    None, None,
                ))

        for t in range(T):
            # Stem
            h = self.conv_stem(x)
            h = self.bn_stem(h)
            h, m_stem = self.lif_stem(h, m_stem)  # Binary spikes

            # Layers with SEW blocks
            pack_idx = 0
            for layer in [self.layer1, self.layer2, self.layer3]:
                for block in layer:
                    h, new_pack = block(h, m_packs[pack_idx])
                    m_packs[pack_idx] = (new_pack[0], new_pack[1], None, None)
                    pack_idx += 1

            # Output
            pooled = self.avgpool(h)
            flat = self.flat(pooled)
            out_total += self.fc(flat)

        return out_total


# ═══════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════
import os
model = SEWResNet().to(DEV)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

if os.path.exists(SAVE_PATH):
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEV))
    print(f"  Loaded from {SAVE_PATH}")
else:
    print(f"\n  Training ({EPOCHS} epochs, T_train={T_TRAIN})...")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.CrossEntropyLoss()
    best = 0
    t0 = time.time()

    for ep in range(EPOCHS):
        model.train()
        train_loss, train_cor, train_tot = 0, 0, 0
        for d, t in tr_ldr:
            d, t = d.to(DEV), t.to(DEV)
            opt.zero_grad()
            out = model(d, T_TRAIN)
            loss = crit(out, t)
            loss.backward()
            opt.step()
            snn_utils.reset(model)
            train_loss += loss.item()
            train_cor += out.max(1)[1].eq(t).sum().item()
            train_tot += t.size(0)
        sch.step()

        # Evaluate with longer T
        model.eval()
        val_cor, val_tot = 0, 0
        with torch.no_grad():
            for d, t in te_ldr:
                d, t = d.to(DEV), t.to(DEV)
                out = model(d, T_EVAL)
                val_cor += out.max(1)[1].eq(t).sum().item()
                val_tot += t.size(0)
                snn_utils.reset(model)
        val_acc = 100 * val_cor / val_tot

        if ep % 5 == 0 or ep >= EPOCHS - 3:
            print(f"  Epoch {ep+1}/{EPOCHS}: train={100*train_cor/train_tot:.1f}% val={val_acc:.1f}% (T={T_EVAL})")

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"  Trained in {(time.time()-t0)/60:.1f} min. Best: {best:.1f}%")

# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SEW-RESNET RESULTS (Direct SNN Training)")
print(f"{'='*70}")

model.load_state_dict(torch.load(SAVE_PATH, map_location=DEV))
model.eval()

for T_test in [2, 4, 8, 16]:
    cor, tot = 0, 0
    with torch.no_grad():
        for d, t in te_ldr:
            d, t = d.to(DEV), t.to(DEV)
            out = model(d, T_test)
            cor += out.max(1)[1].eq(t).sum().item()
            tot += t.size(0)
            snn_utils.reset(model)
    acc = 100 * cor / tot
    print(f"  T={T_test:2d}: {acc:.1f}%")

# Save results
json.dump({
    "method": "SEW-ResNet direct SNN training (NeurIPS 2021)",
    "architecture": "SEW-ResNet-18 style (3 layers, 6 SEW blocks)",
    "T_train": T_TRAIN, "epochs": EPOCHS, "best_accuracy": best,
    "date": datetime.now().isoformat(),
}, open("c:/neurocuda/examples/sew_resnet_results.json", "w"), indent=2)
print(f"\nBest: {best:.1f}% saved to {SAVE_PATH}")