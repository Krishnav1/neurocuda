"""
Debug: Trace ANN→SNN signal flow mismatch in CartPole conversion.
Goal: Understand WHY weight transfer gives 87% not 100%.

Approach: Run ANN and Spiking DQN on the SAME states, compare
layer-by-layer outputs. Find where the mismatch originates.

Usage: python examples/debug_cartpole_gap.py
"""
import sys, os, random
import numpy as np
import torch, torch.nn as nn
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import LIFNeuron, reset_spiking

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ===========================================================================
# Networks (same as demo_b_conversion.py)
# ===========================================================================

class ANNDQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class SpikingDQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=128,
                 beta=0.5, threshold=1.0, T=16):
        super().__init__()
        self.T = T
        self.fc1 = nn.Linear(state_dim, hidden)
        self.lif1 = LIFNeuron(threshold=threshold, beta=beta)
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = LIFNeuron(threshold=threshold, beta=beta)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        B = x.size(0)
        q_accum = torch.zeros(B, self.fc3.out_features, device=x.device)
        for t in range(self.T):
            h = self.lif1(self.fc1(x))
            h = self.lif2(self.fc2(h))
            q_accum = q_accum + self.fc3(h)
        return q_accum / self.T


# ===========================================================================
# Collect states from a trained policy
# ===========================================================================

print("\n[1/4] Collecting states from trained ANN policy...")

# Load or train ANN
ann_model = ANNDQN().to(device)
ckpt_path = "./checkpoints/demo_b_ann_dqn.pt"
if os.path.exists(ckpt_path):
    ann_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"  Loaded ANN checkpoint from {ckpt_path}")
else:
    print("  No checkpoint found — training quick ANN...")
    import gymnasium as gym
    # Quick train (won't be perfect but good enough for debugging)
    env = gym.make("CartPole-v1")
    optimizer = torch.optim.Adam(ann_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    replay = deque(maxlen=10000)
    epsilon = 1.0
    for ep in range(300):
        state, _ = env.reset(); done = False
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = ann_model(s).argmax(1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.append((state, action, reward, next_state, done))
            state = next_state
            if len(replay) >= 64:
                batch = random.sample(replay, 64)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)
                current_q = ann_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = ann_model(next_states).max(1)[0]
                    target_q = rewards + 0.99 * next_q * (1 - dones)
                loss = criterion(current_q, target_q)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(ann_model.parameters(), 1.0)
                optimizer.step()
        epsilon = max(0.01, epsilon * 0.995)
        if ep == 299:
            torch.save(ann_model.state_dict(), ckpt_path)
    env.close()
    print("  Trained quick ANN (300 episodes)")

ann_model.eval()

# Collect diverse states by running the policy
import gymnasium as gym
env = gym.make("CartPole-v1")
states_list = []
state, _ = env.reset()
for _ in range(500):
    states_list.append(state)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = ann_model(s).argmax(1).item()
    state, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        state, _ = env.reset()
env.close()

states_tensor = torch.tensor(np.array(states_list), dtype=torch.float32, device=device)
print(f"  Collected {len(states_tensor)} states")

# ===========================================================================
# Build Spiking DQN with transferred weights
# ===========================================================================

print("\n[2/4] Building Spiking DQN with transferred weights...")

# Test different beta and threshold combinations
configs = [
    {"beta": 0.5, "threshold": 1.0, "label": "v1 (beta=0.5, thr=1.0)"},
    {"beta": 0.0, "threshold": 1.0, "label": "beta=0.0 (no leak)"},
    {"beta": 0.9, "threshold": 1.0, "label": "beta=0.9 (high leak)"},
    {"beta": 0.5, "threshold": 0.5, "label": "thr=0.5 (easy fire)"},
    {"beta": 0.5, "threshold": 2.0, "label": "thr=2.0 (hard fire)"},
]

# ===========================================================================
# Compare ANN vs Spiking layer-by-layer
# ===========================================================================

print("\n[3/4] Comparing ANN vs Spiking DQN layer-by-layer...")
print("=" * 75)

# First, get ANN intermediate outputs
with torch.no_grad():
    ann_h1 = torch.relu(ann_model.fc1(states_tensor))  # (N, 128)
    ann_h2 = torch.relu(ann_model.fc2(ann_h1))          # (N, 128)
    ann_q = ann_model.fc3(ann_h2)                        # (N, 2)

print(f"ANN outputs:")
print(f"  fc1→ReLU:  mean={ann_h1.mean():.3f}, std={ann_h1.std():.3f}, "
      f"sparsity={100*(ann_h1==0).float().mean():.1f}%")
print(f"  fc2→ReLU:  mean={ann_h2.mean():.3f}, std={ann_h2.std():.3f}, "
      f"sparsity={100*(ann_h2==0).float().mean():.1f}%")
print(f"  fc3→Q:     mean={ann_q.mean():.3f}, std={ann_q.std():.3f}")
print(f"  Q range:   [{ann_q.min():.3f}, {ann_q.max():.3f}]")

# Now compare with Spiking DQN for each config
print(f"\n{'Config':<35s} {'LIF1 mean':>10s} {'LIF2 mean':>10s} "
      f"{'Q mean':>10s} {'Q corr':>10s} {'Q err':>10s}")

for cfg in configs:
    snn = SpikingDQN(beta=cfg["beta"], threshold=cfg["threshold"], T=16).to(device)
    snn.fc1.load_state_dict(ann_model.fc1.state_dict())
    snn.fc2.load_state_dict(ann_model.fc2.state_dict())
    snn.fc3.load_state_dict(ann_model.fc3.state_dict())
    snn.eval()

    # Get intermediate outputs from Spiking DQN
    # We need to instrument the forward pass
    B = states_tensor.size(0)
    with torch.no_grad():
        # Manually trace through each layer
        reset_spiking(snn)

        # Layer 1: fc1 + LIF1 (accumulate over T)
        lif1_sum = torch.zeros(B, 128, device=device)
        lif2_sum = torch.zeros(B, 128, device=device)
        for t in range(16):
            h1 = snn.lif1(snn.fc1(states_tensor))
            lif1_sum += h1
            h2 = snn.lif2(snn.fc2(h1))
            lif2_sum += h2

        lif1_avg = lif1_sum / 16  # Average LIF1 output
        lif2_avg = lif2_sum / 16  # Average LIF2 output
        q_avg = snn.fc3(lif2_avg)  # Final Q (fc3 is linear)

    # Compare
    # LIF1 output vs ANN ReLU1 output
    lif1_corr = torch.corrcoef(torch.stack([lif1_avg.flatten(), ann_h1.flatten()]))[0, 1].item()
    lif2_corr = torch.corrcoef(torch.stack([lif2_avg.flatten(), ann_h2.flatten()]))[0, 1].item()
    q_corr = torch.corrcoef(torch.stack([q_avg.flatten(), ann_q.flatten()]))[0, 1].item()
    q_err = (q_avg - ann_q).abs().mean().item()

    print(f"{cfg['label']:<35s} {lif1_avg.mean():>10.3f} {lif2_avg.mean():>10.3f} "
          f"{q_avg.mean():>10.3f} {q_corr:>10.3f} {q_err:>10.3f}")

    # Store Q correlation for best config
    cfg["q_corr"] = q_corr
    cfg["q_err"] = q_err

# ===========================================================================
# Deep Dive: What determines LIF output?
# ===========================================================================

print(f"\n[4/4] Deep dive: LIF dynamics analysis")
print("=" * 75)

# Focus on the best config
best_cfg = max(configs, key=lambda c: c.get("q_corr", 0))
print(f"\nBest config: {best_cfg['label']} (Q correlation = {best_cfg['q_corr']:.4f})")

# Create a fresh Spiking DQN with best config
beta = best_cfg["beta"]
thr = best_cfg["threshold"]
snn = SpikingDQN(beta=beta, threshold=thr, T=16).to(device)
snn.fc1.load_state_dict(ann_model.fc1.state_dict())
snn.fc2.load_state_dict(ann_model.fc2.state_dict())
snn.fc3.load_state_dict(ann_model.fc3.state_dict())
snn.eval()

# Analyze LIF1: what's the input distribution, firing rate?
with torch.no_grad():
    fc1_out = snn.fc1(states_tensor)  # Pre-LIF linear output (N, 128)

print(f"\n  LIF1 analysis:")
print(f"    fc1 output:    mean={fc1_out.mean():.3f}, std={fc1_out.std():.3f}")
print(f"    fc1 range:     [{fc1_out.min():.3f}, {fc1_out.max():.3f}]")

# For each neuron, what's the firing rate over T=16?
reset_spiking(snn)
spike_counts = torch.zeros(1, 128, device=device)
for t in range(16):
    spikes = snn.lif1(snn.fc1(states_tensor[:1]))  # First state, T times
    spike_counts += (spikes > 0).float()
firing_rate = spike_counts / 16

print(f"    Firing rate (1 state, T=16):")
print(f"      Mean firing rate: {firing_rate.mean():.2f} ({firing_rate.mean()*100:.0f}%)")
print(f"      Min/Max:          {firing_rate.min():.2f}/{firing_rate.max():.2f}")
print(f"      Always-on neurons:  {(firing_rate == 1.0).sum().item()}/128")
print(f"      Always-off neurons: {(firing_rate == 0.0).sum().item()}/128")

# What fraction of the fc1 output range is above threshold?
above_thr = (fc1_out.abs() > thr).float().mean()
print(f"    Fraction |fc1| > {thr}: {above_thr:.3f} ({above_thr*100:.1f}%)")

# For a neuron that fires: what does the membrane look like?
print(f"\n  Membrane dynamics (1 neuron, 1 state, T=16):")
reset_spiking(snn)
# Pick a neuron with mid-range fc1 output
fc1_sample = snn.fc1(states_tensor[:1])[0]  # (128,)
mid_idx = fc1_sample.abs().argsort(descending=False)[len(fc1_sample)//2]  # median neuron
fc1_val = fc1_sample[mid_idx].item()
print(f"    Neuron {mid_idx}: fc1 output = {fc1_val:.3f}")
print(f"    t     membrane    spike")
v_trace = []
for t in range(16):
    # Need to step manually
    if snn.lif1.v is None:
        snn.lif1.v = torch.zeros_like(fc1_sample.unsqueeze(0))
    snn.lif1.v = beta * snn.lif1.v + fc1_sample.unsqueeze(0)
    spike = (snn.lif1.v >= thr).float()
    v_val = snn.lif1.v[0, mid_idx].item()
    s_val = spike[0, mid_idx].item()
    v_trace.append(v_val)
    snn.lif1.v = snn.lif1.v - spike * thr
    if t < 8 or t >= 14:
        print(f"    t={t:>2d}   {v_val:>10.4f}   {s_val:.0f}")
    elif t == 8:
        print(f"    ...")

print(f"\n  Membrane trace: {[f'{v:.2f}' for v in v_trace]}")

# ===========================================================================
# The fundamental mismatch
# ===========================================================================

print(f"\n{'='*75}")
print("ROOT CAUSE ANALYSIS")
print("=" * 75)

# Compare: What does ReLU do vs LIF?
# ReLU(x) = max(0, x)  — continuous, unbounded above
# LIF(x) over T timesteps:
#   - Each timestep: binary 0 or threshold
#   - Average over T: threshold * (spike_count / T)
#   - Max possible output: threshold (always fires)
#   - Min possible output: 0 (never fires)

# The key insight:
# For the SAME input x applied T times:
# - With beta=0, v[t] = x each time, spike = (x >= thr)
#   -> always fires or never fires. Output is 0 or thr.
# - With beta>0, v accumulates: v[t] = beta*v[t-1] + x
#   -> v reaches steady state where beta*v + x determines firing

# The steady state for beta=0.5 with fixed input x:
# v[t] = 0.5 * v[t-1] + x
# If neuron fires at threshold: v[t] = 0.5*(v[t-1] - thr) + x
# This is a dynamical system — NOT a simple function of x.

# The root cause: ReLU is a static function. LIF is a dynamical system.
# With the SAME input repeated T times, the LIF output is NOT a function
# of the input alone — it depends on the full temporal dynamics.

# The fix is NOT better thresholds or beta. The fix is BPTT fine-tuning,
# which adapts the fc weights to produce correct Q-values given the
# LIF dynamics. The question is: how much fine-tuning is needed?

print(f"""
  ReLU:  stateless, continuous  →  output = max(0, input)
  LIF:   stateful, binary       →  output = thr * (spikes/T)

  With identical inputs over T timesteps:
    - LIF with beta<1 reaches a limit cycle
    - Output is capped at threshold (not unbounded like ReLU)
    - The mapping input→average_output is sigmoid-like, not linear

  This is why direct weight transfer fails.
  BPTT fine-tuning adapts weights to the LIF transfer function.
  More FT episodes = better adaptation.

  Key metric for FT quality:
    Q-value correlation between ANN and pre-FT SNN
    Best config: {best_cfg['label']}
    Q correlation before FT: {best_cfg['q_corr']:.4f}
    Q error before FT:       {best_cfg['q_err']:.4f}

  With perfect correlation (1.0), weight transfer would preserve
  the policy exactly. The gap (1.0 - {best_cfg['q_corr']:.4f} = {1.0 - best_cfg['q_corr']:.4f})
  must be closed by fine-tuning.
""")

# Find: what's the best achievable Q correlation by tuning beta+thr?
best = max(configs, key=lambda c: c.get("q_corr", 0))
print(f"  Best achievable Q correlation (no FT): {best['q_corr']:.4f} "
      f"with {best['label']}")
print(f"  This means {(1-best['q_corr'])*100:.0f}% of Q-value variance MUST be "
      f"corrected by fine-tuning.")
print(f"  v1 (87% solved) used {best_cfg['label'] if 'v1' in best_cfg['label'] else 'beta=0.5, thr=1.0'} "
      f"— the gap to 100% is the remaining uncorrected variance.")
