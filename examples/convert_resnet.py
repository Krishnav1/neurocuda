"""Convert trained ResNet (90.1%) to SNN and validate."""
import torch, torch.nn as nn, snntorch as snn, numpy as np
from snntorch import surrogate, utils as snn_utils
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

B, T_SNN = 256, 64; DEV = 'cuda'
te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
tr_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
te_ds = datasets.CIFAR10('c:/neurocuda/data', train=False, download=False, transform=te_tf)
tr_ds = datasets.CIFAR10('c:/neurocuda/data', train=True, download=False, transform=tr_tf)

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.c1=nn.Conv2d(c_in,c_out,3,stride=stride,padding=1,bias=False);self.b1=nn.BatchNorm2d(c_out)
        self.c2=nn.Conv2d(c_out,c_out,3,stride=1,padding=1,bias=False);self.b2=nn.BatchNorm2d(c_out)
        self.shortcut=nn.Sequential()
        if stride!=1 or c_in!=c_out: self.shortcut=nn.Sequential(nn.Conv2d(c_in,c_out,1,stride=stride,bias=False),nn.BatchNorm2d(c_out))
        self.relu=nn.ReLU()
    def forward(self,x):
        out=self.relu(self.b1(self.c1(x)));out=self.b2(self.c2(out));out+=self.shortcut(x);return self.relu(out)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False);self.b1=nn.BatchNorm2d(64)
        self.layer1=ResBlock(64,128,stride=2);self.layer2=ResBlock(128,256,stride=2)
        self.layer3=ResBlock(256,512,stride=2);self.avg=nn.AdaptiveAvgPool2d(1);self.flat=nn.Flatten()
        self.fc=nn.Linear(512,10);self.relu=nn.ReLU()
    def forward(self,x):
        x=self.relu(self.b1(self.c1(x)));x=self.layer1(x);x=self.layer2(x);x=self.layer3(x)
        return self.fc(self.flat(self.avg(x)))

ann=ResNet();ann.load_state_dict(torch.load('c:/neurocuda/examples/resnet_cifar10_best.pt',map_location=DEV));ann=ann.to(DEV);ann.eval()

# Fold BN
def fold(conv,bn):
    if conv.bias is None:conv.bias=nn.Parameter(torch.zeros(conv.out_channels))
    s=bn.weight/torch.sqrt(bn.running_var+bn.eps)
    conv.weight.data*=s.view(-1,1,1,1);conv.bias.data=bn.bias-bn.weight*bn.running_mean/torch.sqrt(bn.running_var+bn.eps)
    bn.weight.data=torch.ones_like(bn.weight);bn.bias.data.zero_();bn.running_mean.zero_();bn.running_var.fill_(1.0-bn.eps)

for name,module in ann.named_modules():
    if isinstance(module,nn.BatchNorm2d):
        parent=dict(ann.named_modules());prev=None
        for n,m in parent.items():
            if n==name:break
            if isinstance(m,nn.Conv2d):prev=n
        if prev:fold(dict(ann.named_modules())[prev],module)

# Calibrate
act_data={};handles=[]
def hook_fn(name):
    def hook(module,input,output):
        if name not in act_data:act_data[name]=[]
        act_data[name].append(output.detach().flatten().cpu().numpy())
    return hook
for name,mod in ann.named_modules():
    if isinstance(mod,nn.ReLU):handles.append(mod.register_forward_hook(hook_fn(name)))
with torch.no_grad():
    for d,_ in DataLoader(Subset(tr_ds,range(5000)),B,shuffle=False,drop_last=True):ann(d.to(DEV))
for h in handles:h.remove()

th={}
for name,acts in act_data.items():
    all_vals=np.concatenate(acts);th[name]=max(float(np.percentile(all_vals,95.0)),0.01)
th_list=list(th.values())
print(f'Calibrated {len(th_list)} ReLU layers')

sg=surrogate.fast_sigmoid(slope=25)

class SpikingResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=True);self.c1.load_state_dict({k:v for k,v in ann.c1.state_dict().items()})
        self.l1=snn.Leaky(beta=1.0,threshold=th_list[0],spike_grad=sg,reset_mechanism='subtract')
        self.l1_c1=nn.Conv2d(64,128,3,stride=2,padding=1,bias=True);self.l1_c1.load_state_dict({k:v for k,v in ann.layer1.c1.state_dict().items()})
        self.l1_l1=snn.Leaky(beta=1.0,threshold=th_list[1],spike_grad=sg,reset_mechanism='subtract')
        self.l1_c2=nn.Conv2d(128,128,3,stride=1,padding=1,bias=True);self.l1_c2.load_state_dict({k:v for k,v in ann.layer1.c2.state_dict().items()})
        self.l1_l2=snn.Leaky(beta=1.0,threshold=th_list[2],spike_grad=sg,reset_mechanism='subtract')
        self.l1_sc=nn.Conv2d(64,128,1,stride=2,bias=True);self.l1_sc.load_state_dict({k:v for k,v in ann.layer1.shortcut[0].state_dict().items()})
        self.l2_c1=nn.Conv2d(128,256,3,stride=2,padding=1,bias=True);self.l2_c1.load_state_dict({k:v for k,v in ann.layer2.c1.state_dict().items()})
        self.l2_l1=snn.Leaky(beta=1.0,threshold=th_list[3],spike_grad=sg,reset_mechanism='subtract')
        self.l2_c2=nn.Conv2d(256,256,3,stride=1,padding=1,bias=True);self.l2_c2.load_state_dict({k:v for k,v in ann.layer2.c2.state_dict().items()})
        self.l2_l2=snn.Leaky(beta=1.0,threshold=th_list[4],spike_grad=sg,reset_mechanism='subtract')
        self.l2_sc=nn.Conv2d(128,256,1,stride=2,bias=False);self.l2_sc.load_state_dict({k:v for k,v in ann.layer2.shortcut[0].state_dict().items()})
        self.l3_c1=nn.Conv2d(256,512,3,stride=2,padding=1,bias=True);self.l3_c1.load_state_dict({k:v for k,v in ann.layer3.c1.state_dict().items()})
        self.l3_l1=snn.Leaky(beta=1.0,threshold=th_list[5],spike_grad=sg,reset_mechanism='subtract')
        self.l3_c2=nn.Conv2d(512,512,3,stride=1,padding=1,bias=True);self.l3_c2.load_state_dict({k:v for k,v in ann.layer3.c2.state_dict().items()})
        self.l3_l2=snn.Leaky(beta=1.0,threshold=th_list[6],spike_grad=sg,reset_mechanism='subtract')
        self.l3_sc=nn.Conv2d(256,512,1,stride=2,bias=False);self.l3_sc.load_state_dict({k:v for k,v in ann.layer3.shortcut[0].state_dict().items()})
        self.avg=nn.AdaptiveAvgPool2d(1);self.flat=nn.Flatten();self.fc=nn.Linear(512,10)
        self.fc.load_state_dict({k:v for k,v in ann.fc.state_dict().items()})
    def forward(self,x):
        m=[l.init_leaky() for l in [self.l1,self.l1_l1,self.l1_l2,self.l2_l1,self.l2_l2,self.l3_l1,self.l3_l2]]
        out=torch.zeros(x.size(0),10,device=x.device)
        for _ in range(T_SNN):
            s0,m[0]=self.l1(torch.relu(self.c1(x)),m[0])
            r1=self.l1_sc(x);c1a,m[1]=self.l1_l1(torch.relu(self.l1_c1(s0)),m[1]);c1b,m[2]=self.l1_l2(torch.relu(self.l1_c2(c1a)),m[2]);s1=c1b+r1
            r2=self.l2_sc(s1);c2a,m[3]=self.l2_l1(torch.relu(self.l2_c1(s1)),m[3]);c2b,m[4]=self.l2_l2(torch.relu(self.l2_c2(c2a)),m[4]);s2=c2b+r2
            r3=self.l3_sc(s2);c3a,m[5]=self.l3_l1(torch.relu(self.l3_c1(s2)),m[5]);c3b,m[6]=self.l3_l2(torch.relu(self.l3_c2(c3a)),m[6]);s3=c3b+r3
            out+=self.fc(self.flat(self.avg(s3)))
        return out

test_ldr=DataLoader(Subset(te_ds,range(1000)),B,shuffle=False,drop_last=True)
cor,tot=0,0
with torch.no_grad():
    for d,t in test_ldr:d,t=d.to(DEV),t.to(DEV);cor+=ann(d).max(1)[1].eq(t).sum().item();tot+=t.size(0)
ann_acc=100*cor/tot;print(f'ANN: {ann_acc:.1f}%')

snn_resnet=SpikingResNet().to(DEV);snn_resnet.eval()
cor,tot=0,0
with torch.no_grad():
    for d,t in test_ldr:d,t=d.to(DEV),t.to(DEV);cor+=snn_resnet(d).max(1)[1].eq(t).sum().item();tot+=t.size(0)
snn_acc=100*cor/tot;print(f'SNN (converted): {snn_acc:.1f}% | Gap: {ann_acc-snn_acc:.1f}%')

# Fine-tune
snn_ft=SpikingResNet().to(DEV)
ft_ldr=DataLoader(Subset(tr_ds,range(10000)),B,shuffle=True,drop_last=True)
for ep in range(3):
    lr=[1e-5,5e-6,1e-6][ep];opt=torch.optim.AdamW(snn_ft.parameters(),lr=lr);crit=nn.CrossEntropyLoss();snn_ft.train()
    for d,t in ft_ldr:d,t=d.to(DEV),t.to(DEV);opt.zero_grad();crit(snn_ft(d),t).backward();opt.step();snn_utils.reset(snn_ft)
    snn_ft.eval();cor,tot=0,0
    with torch.no_grad():
        for d,t in test_ldr:d,t=d.to(DEV),t.to(DEV);cor+=snn_ft(d).max(1)[1].eq(t).sum().item();tot+=t.size(0)
    print(f'  FT epoch {ep+1}: {100*cor/tot:.1f}%')

snn_ft.eval();cor,tot=0,0
with torch.no_grad():
    for d,t in test_ldr:d,t=d.to(DEV),t.to(DEV);cor+=snn_ft(d).max(1)[1].eq(t).sum().item();tot+=t.size(0)
ft_acc=100*cor/tot
print(f'\nRESNET: ANN={ann_acc:.1f}% | SNN={snn_acc:.1f}% | SNN-FT={ft_acc:.1f}% | Gap={ann_acc-ft_acc:.1f}%')
