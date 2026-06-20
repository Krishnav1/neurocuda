"""
Demo B v2 — CartPole ANN→SNN Conversion (Improved Hyperparameters)
====================================================================
Improvements over demo_b_conversion.py (87% solved):
  1. Longer fine-tuning: 500 episodes (was 300)
  2. Better epsilon schedule: start 0.7, slower decay
  3. Target network updates every 200 steps (was 100)
  4. Gradient clipping at 0.5 (was 1.0)

Key insight: BPTT fine-tuning adapts weights to the LIF transfer function.
The ANN weights are just a good initialization — longer FT closes the gap.

Pipeline:
  1. Train ANN DQN (ReLU) — same as v1
  2. Transfer fc weights to Spiking DQN (LIF, beta=0.5, thresh=1.0)
  3. Fine-tune with BPTT (500 episodes, tuned schedule)
  4. Evaluate

Usage: python examples/demo_b_conversion_v2.py
"""
import sys, os, time, random
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
        return self.fc3(x)  # No activation after fc3 (Q-values)


class SpikingDQN(nn.Module):
    """Spiking DQN — rate-coded Q-values over T timesteps."""
    def __init__(self, state_dim=4, action_dim=2, hidden=128,
                 beta=0.9, threshold=1.0, T=16):
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
# Replay Buffer + Training
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


def train_dqn(model_factory, is_spiking, n_episodes, lr, label=""):
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    policy_net = model_factory().to(device)
    target_net = model_factory().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    replay = ReplayBuffer()

    epsilon = 1.0
    steps_done = 0
    rewards_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    if is_spiking: reset_spiking(policy_net)
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

                if is_spiking: reset_spiking(policy_net)
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    if is_spiking: reset_spiking(target_net)
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + 0.99 * next_q * (1 - dones_r)

                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            if steps_done % 100 == 0:
                target_net.load_state_dict(policy_net.state_dict())

        rewards_history.append(episode_reward)
        epsilon = max(0.01, epsilon * 0.995)

        avg_100 = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
        if (episode + 1) % 100 == 0:
            print(f"  {label} Ep {episode+1}: Avg100 = {avg_100:.1f}, eps = {epsilon:.3f}")
        if len(rewards_history) >= 100 and avg_100 >= 195:
            print(f"  {label} Solved at episode {episode+1}! (avg100 = {avg_100:.1f})")
            break

    env.close()
    return policy_net, rewards_history


def evaluate(model, is_spiking, n_episodes=100):
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
# Main
# ===========================================================================

if __name__ == "__main__":
    print(f"Device: {device}")
    print("CartPole ANN→SNN Conversion v2 — Better Fine-Tuning Schedule")
    print("=" * 60)
    t_start = time.time()

    import gymnasium as gym

    # --- 1. Train ANN DQN ---
    print("\nSTEP 1: Train ANN DQN (ReLU)")
    print("-" * 40)
    ann_model, ann_hist = train_dqn(
        lambda: ANNDQN(), is_spiking=False, n_episodes=600, lr=1e-3, label="ANN"
    )
    ann_mean, ann_solved = evaluate(ann_model, is_spiking=False)
    print(f"  ANN DQN: Mean Reward = {ann_mean:.1f}, Solved = {ann_solved:.0f}%")
    torch.save(ann_model.state_dict(), "./checkpoints/demo_b_ann_dqn_v2.pt")

    # --- 2. Create Spiking DQN + Weight Transfer ---
    print("\nSTEP 2: Create Spiking DQN + Transfer ANN Weights")
    print("-" * 40)
    snn_model = SpikingDQN(T=16, beta=0.5, threshold=1.0).to(device)

    # Transfer fc weights from ANN
    snn_model.fc1.load_state_dict(ann_model.fc1.state_dict())
    snn_model.fc2.load_state_dict(ann_model.fc2.state_dict())
    snn_model.fc3.load_state_dict(ann_model.fc3.state_dict())
    print("  fc1, fc2, fc3 weights transferred from ANN DQN")
    print("  LIF: beta=0.5, threshold=1.0 (same as direct SNN training)")

    # Evaluate BEFORE fine-tuning
    snn_mean_before, snn_solved_before = evaluate(snn_model, is_spiking=True)
    print(f"  Spiking DQN BEFORE fine-tune: Mean = {snn_mean_before:.1f}, "
          f"Solved = {snn_solved_before:.0f}%")

    # --- 3. Fine-tune Spiking DQN ---
    print("\nSTEP 3: Fine-Tune Spiking DQN (BPTT + Surrogate Gradient)")
    print("-" * 40)
    print("  Improvements vs v1: 500 eps, epsilon 0.7→0.05, target update every 200")

    env_ft = gym.make("CartPole-v1")
    target_net = SpikingDQN(T=16, beta=0.5, threshold=1.0).to(device)
    target_net.load_state_dict(snn_model.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(snn_model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    replay = ReplayBuffer()
    epsilon = 0.7  # Higher start → more exploration early
    steps_done = 0
    ft_rewards = []

    for episode in range(500):
        state, _ = env_ft.reset(); episode_reward = 0; done = False
        while not done:
            if random.random() < epsilon:
                action = env_ft.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    reset_spiking(snn_model)
                    action = snn_model(s).argmax(1).item()

            next_state, reward, terminated, truncated, _ = env_ft.step(action)
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
                torch.nn.utils.clip_grad_norm_(snn_model.parameters(), 0.5)  # Tighter clip
                optimizer.step()

                if steps_done % 200 == 0:  # Less frequent → more stable
                    target_net.load_state_dict(snn_model.state_dict())

        ft_rewards.append(episode_reward)
        epsilon = max(0.05, epsilon * 0.993)  # Slower decay

        avg_100 = np.mean(ft_rewards[-100:]) if len(ft_rewards) >= 100 else np.mean(ft_rewards)
        if (episode + 1) % 50 == 0:
            print(f"  FT Ep {episode+1}: Avg100 = {avg_100:.1f}, eps = {epsilon:.3f}")
        if len(ft_rewards) >= 100 and avg_100 >= 195:
            print(f"  Solved at fine-tune episode {episode+1}! (avg100 = {avg_100:.1f})")
            break
    env_ft.close()

    # --- 4. Evaluate ---
    print("\nSTEP 4: Final Evaluation")
    print("-" * 40)
    snn_model.eval()
    snn_mean, snn_solved = evaluate(snn_model, is_spiking=True, n_episodes=100)
    print(f"  Spiking DQN AFTER fine-tune: Mean = {snn_mean:.1f}, Solved = {snn_solved:.0f}%")

    # --- 5. Sparsity ---
    print("\nSTEP 5: Spiking Sparsity")
    print("-" * 40)
    sparsity, spikes, total, layer_data = measure_sparsity(snn_model)
    print(f"  Spiking Sparsity: {sparsity:.2f}%")
    for name, d in sorted(layer_data.items()):
        ls = 100.0 * (1.0 - d["spikes"] / max(d["total"], 1))
        print(f"    {name}: {ls:.2f}% ({d['spikes']:,}/{d['total']:,})")

    # --- Summary ---
    improvement = snn_solved - 87.0  # vs v1
    print(f"\n{'='*60}")
    print("CART-POLE CONVERSION v2 — SUMMARY")
    print("=" * 60)
    print(f"""
  Method:         Weight transfer + BPTT fine-tune (improved schedule)
  ANN DQN:        Mean Reward = {ann_mean:.1f}, Solved = {ann_solved:.0f}%
  Spiking DQN:
    Before FT:    Mean Reward = {snn_mean_before:.1f}, Solved = {snn_solved_before:.0f}%
    After FT:     Mean Reward = {snn_mean:.1f}, Solved = {snn_solved:.0f}%
  Sparsity:       {sparsity:.2f}%
  vs v1 (87%):    {improvement:+.0f}% points
  Total time:     {(time.time() - t_start) / 60:.1f} min
""")
    print("=" * 60)
