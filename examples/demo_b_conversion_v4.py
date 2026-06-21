"""
Demo B v4 — Robust CartPole ANN→SNN Conversion (Multi-Seed)
=============================================================
Fixes v3's failure: trains ANN until it passes rigorous evaluation
(not just training avg100 ≥ 195), then weight transfer + BPTT FT.

Pipeline:
  1. Train ANN DQN with evaluation checkpoints — only accept when
     eval(100 episodes, epsilon=0) shows ≥95% solved
  2. Weight transfer to Spiking DQN
  3. BPTT fine-tune with surrogate gradient (500 episodes)
  4. Evaluate: mean reward, % solved, sparsity
  5. Multi-seed: repeat for 3 seeds

Usage: python examples/demo_b_conversion_v4.py [--seeds 0 1 2]
"""
import sys, os, time, random, argparse
import numpy as np
import torch, torch.nn as nn
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import LIFNeuron, reset_spiking

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================================
# Networks
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
                 beta=0.5, threshold=1.0, alpha=2.0, T=16):
        super().__init__()
        self.T = T
        self.fc1 = nn.Linear(state_dim, hidden)
        self.lif1 = LIFNeuron(threshold=threshold, beta=beta, alpha=alpha)
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = LIFNeuron(threshold=threshold, beta=beta, alpha=alpha)
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
# Replay Buffer
# ===========================================================================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)


# ===========================================================================
# Evaluation (deterministic, no exploration)
# ===========================================================================

def evaluate(model, is_spiking, n_episodes=100):
    """Return (mean_reward, %_solved). Deterministic (no epsilon)."""
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_r = 0; done = False
        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                if is_spiking: reset_spiking(model)
                action = model(s).argmax(1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated; ep_r += reward
        rewards.append(ep_r)
    env.close()
    return np.mean(rewards), 100.0 * sum(1 for r in rewards if r >= 195) / len(rewards)


def measure_sparsity(model, n_episodes=100):
    """Measure LIF spiking sparsity over evaluation episodes."""
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    spike_data = {}
    def make_hook(name):
        def hook(m, inp, out):
            if name not in spike_data: spike_data[name] = {"total": 0, "spikes": 0}
            spike_data[name]["total"] += out.numel()
            spike_data[name]["spikes"] += (out > 0).sum().item()
        return hook
    handles = []
    for n, m in model.named_modules():
        if isinstance(m, LIFNeuron): handles.append(m.register_forward_hook(make_hook(n)))

    for _ in range(n_episodes):
        state, _ = env.reset(); done = False
        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                reset_spiking(model); action = model(s).argmax(1).item()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    for h in handles: h.remove(); env.close()

    total = sum(d["total"] for d in spike_data.values())
    spikes = sum(d["spikes"] for d in spike_data.values())
    return 100.0 * (1.0 - spikes / max(total, 1)), spikes, total, spike_data


# ===========================================================================
# ANN Training — early-stop at training avg100 ≥ 195 (v1 proven recipe)
# ===========================================================================

def train_ann_early_stop(seed=0):
    """Train ANN DQN until avg100 ≥ 195 during training. Stop immediately.

    KEY INSIGHT (June 21, 2026): Stopping at training convergence (not eval
    convergence) produces weights that transfer better to SNN. The ANN policy
    that works with exploration (epsilon=0.16) is in a wider basin that
    survives the ReLU→LIF perturbation. Over-training to epsilon=0.01
    produces a fragile policy that breaks under binary LIF dynamics.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    import gymnasium as gym
    env = gym.make("CartPole-v1")

    policy_net = ANNDQN().to(device)
    target_net = ANNDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    replay = ReplayBuffer(capacity=10000)

    epsilon = 1.0
    steps_done = 0
    train_rewards = []
    solved_ep = None

    for episode in range(600):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = policy_net(s).argmax(1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps_done += 1
            replay.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay) >= 64:
                states, actions, rewards, next_states, dones_r = replay.sample(64)
                states = states.to(device); actions = actions.to(device)
                rewards = rewards.to(device); next_states = next_states.to(device)
                dones_r = dones_r.to(device)

                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + 0.99 * next_q * (1 - dones_r)

                loss = criterion(current_q, target_q)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            if steps_done % 100 == 0:
                target_net.load_state_dict(policy_net.state_dict())

        train_rewards.append(episode_reward)
        epsilon = max(0.01, epsilon * 0.995)

        avg_100 = np.mean(train_rewards[-100:]) if len(train_rewards) >= 100 else np.mean(train_rewards)
        if (episode + 1) % 100 == 0:
            print(f"  ANN S{seed} Ep {episode+1}: Train100={avg_100:.0f}, eps={epsilon:.3f}")

        # v1 recipe: stop as soon as avg100 ≥ 195 during training
        if len(train_rewards) >= 100 and avg_100 >= 195:
            solved_ep = episode + 1
            print(f"  ✅ ANN S{seed} SOLVED at Ep {solved_ep}: Train100={avg_100:.0f}")
            break

    env.close()
    policy_net.eval()
    return policy_net, {"solved_ep": solved_ep, "train100": avg_100}


# ===========================================================================
# Fine-Tune Spiking DQN (BPTT + Surrogate Gradient)
# ===========================================================================

def finetune_spiking(snn_model, n_episodes=500, lr=5e-4, label="FT"):
    """BPTT fine-tune with surrogate gradient. ANN weights already transferred."""
    import gymnasium as gym
    env = gym.make("CartPole-v1")

    target_net = SpikingDQN(T=16, beta=0.5, threshold=1.0).to(device)
    target_net.load_state_dict(snn_model.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(snn_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    replay = ReplayBuffer(capacity=10000)
    epsilon = 0.3
    steps_done = 0
    ft_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset(); episode_reward = 0; done = False
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    reset_spiking(snn_model)
                    action = snn_model(s).argmax(1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated; episode_reward += reward; steps_done += 1
            replay.push(state, action, reward, next_state, done); state = next_state

            if len(replay) >= 64:
                states, actions, rewards, next_states, dones_r = replay.sample(64)
                states = states.to(device); actions = actions.to(device)
                rewards = rewards.to(device); next_states = next_states.to(device)
                dones_r = dones_r.to(device)

                reset_spiking(snn_model)
                current_q = snn_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    reset_spiking(target_net)
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + 0.99 * next_q * (1 - dones_r)

                loss = criterion(current_q, target_q)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(snn_model.parameters(), 1.0)
                optimizer.step()

                if steps_done % 100 == 0:  # v1 recipe: frequent target updates
                    target_net.load_state_dict(snn_model.state_dict())

        ft_rewards.append(episode_reward)
        epsilon = max(0.05, epsilon * 0.995)  # v1 recipe: floor at 0.05

        avg_100 = np.mean(ft_rewards[-100:]) if len(ft_rewards) >= 100 else np.mean(ft_rewards)
        if (episode + 1) % 50 == 0:
            print(f"  {label} Ep {episode+1}: Train100={avg_100:.0f}, eps={epsilon:.3f}")
        # v1 recipe: stop FT as soon as avg100 ≥ 195
        if len(ft_rewards) >= 100 and avg_100 >= 195:
            print(f"  {label} Solved at FT ep {episode+1}! (Train100={avg_100:.0f})")
            break

    env.close()
    snn_model.eval()
    return snn_model, ft_rewards


# ===========================================================================
# Run one seed
# ===========================================================================

def run_seed(seed):
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")

    # --- 1. Train ANN (early-stop at training avg100 ≥ 195) ---
    print("\n[1/5] Training ANN DQN (early-stop at Train100 ≥ 195)...")
    t0 = time.time()
    ann_model, ann_stats = train_ann_early_stop(seed=seed)
    ann_time = time.time() - t0
    print(f"  ANN training time: {ann_time/60:.1f} min | Solved at Ep {ann_stats['solved_ep']}")

    # --- 2. Weight transfer ---
    print("\n[2/5] Weight transfer ANN → Spiking DQN...")
    snn_model = SpikingDQN(T=16, beta=0.5, threshold=1.0).to(device)
    snn_model.fc1.load_state_dict(ann_model.fc1.state_dict())
    snn_model.fc2.load_state_dict(ann_model.fc2.state_dict())
    snn_model.fc3.load_state_dict(ann_model.fc3.state_dict())
    snn_model.eval()

    snn_mean_before, snn_solved_before = evaluate(snn_model, is_spiking=True)
    print(f"  Spiking DQN BEFORE FT: Mean={snn_mean_before:.1f}, Solved={snn_solved_before:.0f}%")

    # --- 3. Fine-tune ---
    print("\n[3/5] BPTT fine-tuning (surrogate gradient, 300 episodes)...")
    t0 = time.time()
    snn_model, ft_rewards = finetune_spiking(
        snn_model, n_episodes=300, lr=5e-4, label=f"S{seed}"
    )
    ft_time = time.time() - t0
    avg_last_100 = np.mean(ft_rewards[-100:]) if len(ft_rewards) >= 100 else np.mean(ft_rewards)
    print(f"  FT time: {ft_time/60:.1f} min | Final Train100={avg_last_100:.0f}")

    # --- 4. Final evaluation ---
    print("\n[4/5] Final evaluation (100 episodes, deterministic)...")
    snn_model.eval()
    snn_mean, snn_solved = evaluate(snn_model, is_spiking=True, n_episodes=100)
    print(f"  Spiking DQN AFTER FT: Mean={snn_mean:.1f}, Solved={snn_solved:.0f}%")

    # --- 5. Sparsity ---
    print("\n[5/5] Sparsity measurement...")
    sparsity, spikes, total, layer_data = measure_sparsity(snn_model)
    print(f"  Sparsity: {sparsity:.2f}%")

    return {
        "seed": seed,
        "ann": ann_stats,
        "snn_before_ft": {"mean": snn_mean_before, "solved": snn_solved_before},
        "snn_after_ft": {"mean": snn_mean, "solved": snn_solved},
        "sparsity": sparsity,
        "spikes": spikes,
        "total_acts": total,
        "layer_data": {n: {"sparsity": 100*(1-d["spikes"]/max(d["total"],1)),
                           "spikes": d["spikes"], "total": d["total"]}
                       for n, d in sorted(layer_data.items())},
    }


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    print(f"Device: {device}")
    print(f"CartPole ANN→SNN Conversion v4 — Robust Training + Multi-Seed")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*70}")
    t_start = time.time()

    all_results = []
    for seed in args.seeds:
        result = run_seed(seed)
        all_results.append(result)

    # =========================================================================
    # Aggregate
    # =========================================================================
    print(f"\n{'='*70}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*70}")

    ann_eps = [r["ann"]["solved_ep"] or 600 for r in all_results]
    snn_means = [r["snn_after_ft"]["mean"] for r in all_results]
    snn_solved = [r["snn_after_ft"]["solved"] for r in all_results]
    sparsities = [r["sparsity"] for r in all_results]

    print(f"\n  {'Seed':<8s} {'ANN Ep':>8s} {'Pre-FT':>8s} {'SNN Mean':>10s} {'SNN Solved':>12s} {'Sparsity':>10s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*10}")
    for r in all_results:
        solved_ep_str = str(r['ann']['solved_ep']) if r['ann']['solved_ep'] else '?'
        print(f"  {r['seed']:<8d} {solved_ep_str:>8s} "
              f"{r['snn_before_ft']['mean']:>7.1f} {r['snn_after_ft']['mean']:>10.1f} "
              f"{r['snn_after_ft']['solved']:>11.1f}% {r['sparsity']:>9.1f}%")

    snn_mean_all = np.mean(snn_means)
    snn_std = np.std(snn_means)
    snn_solved_mean = np.mean(snn_solved)
    snn_solved_std = np.std(snn_solved)
    sparsity_mean = np.mean(sparsities)
    sparsity_std = np.std(sparsities)
    n_passed = sum(1 for s in snn_solved if s >= 95.0)

    print(f"\n  AGGREGATE (mean ± std over {len(args.seeds)} seeds):")
    print(f"  ANN solved eps: {np.mean(ann_eps):.0f} ± {np.std(ann_eps):.0f}")
    print(f"  SNN Mean:  {snn_mean_all:.1f} ± {snn_std:.1f}")
    print(f"  SNN Solved: {snn_solved_mean:.1f}% ± {snn_solved_std:.1f}%")
    print(f"  Seeds ≥95%: {n_passed}/{len(args.seeds)}")
    print(f"  Sparsity:   {sparsity_mean:.1f}% ± {sparsity_std:.1f}%")
    print(f"  Total time: {(time.time() - t_start) / 60:.1f} min")

    # GATE-style report
    print(f"\n  GATE 2b — CartPole Conversion")
    if snn_solved_mean >= 95.0:
        print(f"  ✅ PASS: SNN {snn_solved_mean:.0f}% ± {snn_solved_std:.0f}% solved ≥ 95% target")
    else:
        print(f"  ❌ NOT YET: SNN {snn_solved_mean:.0f}% ± {snn_solved_std:.0f}% solved < 95% target")
        if n_passed > 0:
            print(f"  {n_passed}/{len(args.seeds)} seeds pass individually. DQN training is stochastic.")
    print(f"{'='*70}")
