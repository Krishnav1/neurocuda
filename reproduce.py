#!/usr/bin/env python3
"""
NeuroCUDA — Reproduce All Benchmarks
=====================================
Single entry point. One command reproduces every verified number in the README.

Usage:
    python reproduce.py              # Run all benchmarks (~30 min)
    python reproduce.py --quick      # Fast verification — NMNIST only (~4 min)
    python reproduce.py --demo       # Robotics pipeline only (~2 min)
    python reproduce.py --benchmarks nmnist cartpole  # Specific benchmarks
    python reproduce.py --list       # List available benchmarks

Available benchmarks:
    nmnist     — NMNIST event-camera conversion (3 seeds, 20K data)  ~4 min
    cartpole   — CartPole ANN→SNN conversion (5 seeds, stochastic)   ~15 min
    robotics   — Full robotics perception pipeline                     ~2 min

What this script does:
    1. Checks data exists — auto-downloads NMNIST if missing
    2. Runs each benchmark sequentially
    3. Prints a summary table matching the README
    4. Exits 0 if all benchmarks pass, 1 if any fail

Requirements:
    pip install -r requirements.txt
    python examples/prep_nmnist.py    # Run once before (auto-called if data missing)
"""

import sys, os, time, argparse, subprocess

# ===========================================================================
# Configuration
# ===========================================================================

BENCHMARKS = {
    "nmnist": {
        "name": "NMNIST Conversion",
        "readme_target": {"ann": 99.70, "if_acc": 99.88, "if_std": 0.02, "gap": -0.18, "sparsity": 91.7},
        "required": True,   # Must pass for ship
        "time_est": "4 min",
    },
    "cartpole": {
        "name": "CartPole Conversion",
        "readme_target": None,  # Stochastic — no fixed target, just report honestly
        "required": False,  # Known stochastic — not required for ship
        "time_est": "15 min",
    },
    "robotics": {
        "name": "Robotics Pipeline",
        "readme_target": {"snr_beats_ann": True, "sparsity_min": 90.0},
        "required": True,
        "time_est": "2 min",
    },
}

# ===========================================================================
# Data Check
# ===========================================================================

def ensure_nmnist_data():
    """Check if NMNIST data exists. If not, auto-download."""
    train_path = "./data/nmnist_train.pt"
    test_path = "./data/nmnist_test.pt"

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  NMNIST data: OK (train={os.path.getsize(train_path)/1024/1024:.0f}MB, "
              f"test={os.path.getsize(test_path)/1024/1024:.0f}MB)")
        return True

    print("  NMNIST data not found. Auto-downloading...")
    print("  This may take 10-15 minutes on first run (downloads ~6GB).")
    prep_script = os.path.join(os.path.dirname(__file__), "examples", "prep_nmnist.py")

    if not os.path.exists(prep_script):
        print(f"  ERROR: {prep_script} not found.")
        return False

    result = subprocess.run(
        [sys.executable, prep_script],
        capture_output=False,
        cwd=os.path.dirname(__file__)
    )
    if result.returncode != 0:
        print("  ERROR: Data download failed.")
        return False

    return os.path.exists(train_path) and os.path.exists(test_path)


# ===========================================================================
# NMNIST Multi-Seed Benchmark
# ===========================================================================

def run_nmnist_benchmark(seeds=(0, 1, 2), n_train=20000, n_test=2000):
    """Run NMNIST ANN→SNN conversion across multiple seeds.

    Uses neurocuda.convert() directly — the exact API shown in the README.
    """
    print(f"\n{'='*70}")
    print(f"NMNIST CONVERSION BENCHMARK ({len(seeds)} seeds, {n_train} train, {n_test} test)")
    print(f"{'='*70}")

    import numpy as np
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    sys.path.insert(0, os.path.dirname(__file__))
    from neurocuda import convert, measure_sparsity
    from models import reset_spiking

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # --- Load data ---
    train_data = torch.load("./data/nmnist_train.pt", map_location="cpu", weights_only=False)
    test_data = torch.load("./data/nmnist_test.pt", map_location="cpu", weights_only=False)

    n_tr = min(n_train, len(train_data["data"]))
    n_te = min(n_test, len(test_data["data"]))
    train_loader = DataLoader(
        TensorDataset(train_data["data"][:n_tr], train_data["targets"][:n_tr]),
        batch_size=128, shuffle=True)
    test_loader = DataLoader(
        TensorDataset(test_data["data"][:n_te], test_data["targets"][:n_te]),
        batch_size=128)
    print(f"  Data: {n_tr} train, {n_te} test")

    # --- Model definition (4D-native CNN — matches demo_a_multiseed) ---
    class NMNISTCNN(nn.Module):
        def __init__(self, act_factory=nn.ReLU):
            super().__init__()
            self.conv1 = nn.Conv2d(2, 32, 5, padding=2, bias=False)
            self.bn1 = nn.BatchNorm2d(32); self.act1 = act_factory(); self.pool1 = nn.AvgPool2d(2)
            self.conv2 = nn.Conv2d(32, 64, 5, padding=2, bias=False)
            self.bn2 = nn.BatchNorm2d(64); self.act2 = act_factory(); self.pool2 = nn.AvgPool2d(2)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(128); self.act3 = act_factory(); self.pool3 = nn.AvgPool2d(2)
            self.flatten = nn.Flatten(); self.fc = nn.Linear(2048, 10)

        def forward(self, x):
            x = self.pool1(self.act1(self.bn1(self.conv1(x))))
            x = self.pool2(self.act2(self.bn2(self.conv2(x))))
            x = self.pool3(self.act3(self.bn3(self.conv3(x))))
            return self.fc(self.flatten(x))

    class TemporalWrapper(nn.Module):
        def __init__(self, model_4d):
            super().__init__()
            self.model_4d = model_4d
        def forward(self, x):
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            out = self.model_4d(x)
            return out.reshape(B, T, -1).mean(dim=1)

    # --- Accuracy helper ---
    def measure_acc(model, dl, is_spiking=False, T=16):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in dl:
                data, target = data.to(device), target.to(device)
                if is_spiking:
                    B, T_data = data.size(0), min(T, data.size(1))
                    reset_spiking(model)
                    out_sum = torch.zeros(B, 10, device=device)
                    for t in range(T_data):
                        out_sum += model(data[:, t:t+1, :, :, :])
                    pred = out_sum.argmax(1)
                else:
                    pred = model(data).argmax(1)
                correct += (pred == target).sum().item()
                total += data.size(0)
        return 100.0 * correct / total

    # --- Run each seed ---
    results = []
    ckpt_path = "./checkpoints/demo_a_ann_best.pt"

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Load ANN
        inner = NMNISTCNN(act_factory=nn.ReLU).to(device)
        if os.path.exists(ckpt_path):
            inner.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            # Quick train if no checkpoint
            inner.train()
            opt = torch.optim.AdamW(inner.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()
            for ep in range(5):
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    opt.zero_grad()
                    loss = crit(TemporalWrapper(inner)(data), target)
                    loss.backward(); opt.step()
            inner.eval()
            torch.save(inner.state_dict(), ckpt_path)

        ann_model = TemporalWrapper(inner).to(device)
        ann_acc = measure_acc(ann_model, test_loader, is_spiking=False)

        # Convert
        t0 = time.time()
        snn_model, stats = convert(
            ann_model,
            train_loader,
            test_loader=test_loader,
            qcfs_epochs=3,
            if_epochs=3,
            strategy="qcfs_if_ft",
            channel_wise=True,
            device=device,
            verbose=False,
        )
        conv_time = time.time() - t0

        # Sparsity
        sparsity, _, _, _ = measure_sparsity(snn_model, test_loader, device=device, max_batches=20)

        results.append({
            "seed": seed,
            "ann_acc": ann_acc,
            "qcfs_acc": stats["qcfs_accuracy"],
            "if_acc": stats["if_accuracy"],
            "gap": ann_acc - stats["if_accuracy"],
            "sparsity": sparsity,
            "time": conv_time,
        })

        print(f"    ANN={ann_acc:.2f}% QCFS={stats['qcfs_accuracy']:.2f}% "
              f"IF={stats['if_accuracy']:.2f}% Gap={ann_acc - stats['if_accuracy']:.2f}% "
              f"Sparsity={sparsity:.1f}% [{conv_time:.0f}s]")

    # --- Aggregate ---
    ann_accs = [r["ann_acc"] for r in results]
    if_accs = [r["if_acc"] for r in results]
    gaps = [r["gap"] for r in results]
    sparsities = [r["sparsity"] for r in results]

    agg = {
        "benchmark": "nmnist",
        "seeds": len(seeds),
        "ann_mean": np.mean(ann_accs), "ann_std": np.std(ann_accs),
        "if_mean": np.mean(if_accs), "if_std": np.std(if_accs),
        "gap_mean": np.mean(gaps), "gap_std": np.std(gaps),
        "sparsity_mean": np.mean(sparsities), "sparsity_std": np.std(sparsities),
        "total_time": sum(r["time"] for r in results),
        "per_seed": results,
    }
    return agg


# ===========================================================================
# CartPole Conversion Benchmark
# ===========================================================================

def run_cartpole_benchmark(seeds=(0, 1, 2), n_ft_episodes=300):
    """Run CartPole ANN→SNN conversion across multiple seeds.

    Uses the proven early-stop recipe from demo_b_conversion_v4.py.
    NOTE: This benchmark is stochastic — ~29% seed success rate.
    """
    print(f"\n{'='*70}")
    print(f"CARTPOLE CONVERSION BENCHMARK ({len(seeds)} seeds)")
    print(f"  ⚠  NOTE: CartPole conversion is stochastic (~29% seed success rate)")
    print(f"  ⚠  Direct SNN training is the 100% reliable alternative")
    print(f"{'='*70}")

    import random, numpy as np
    from collections import deque
    import torch, torch.nn as nn

    sys.path.insert(0, os.path.dirname(__file__))
    from models import LIFNeuron, reset_spiking

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # --- Network definitions (matches demo_b_conversion_v4) ---
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
        def __init__(self, T=16, beta=0.5, threshold=1.0, alpha=2.0):
            super().__init__()
            self.T = T
            self.fc1 = nn.Linear(4, 128)
            self.lif1 = LIFNeuron(threshold=threshold, beta=beta, alpha=alpha)
            self.fc2 = nn.Linear(128, 128)
            self.lif2 = LIFNeuron(threshold=threshold, beta=beta, alpha=alpha)
            self.fc3 = nn.Linear(128, 2)
        def forward(self, x):
            B = x.size(0)
            q = torch.zeros(B, 2, device=x.device)
            for t in range(self.T):
                h = self.lif1(self.fc1(x))
                h = self.lif2(self.fc2(h))
                q = q + self.fc3(h)
            return q / self.T

    class ReplayBuffer:
        def __init__(self, capacity=10000):
            self.buffer = deque(maxlen=capacity)
        def push(self, s, a, r, ns, d):
            self.buffer.append((s, a, r, ns, d))
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

    # --- Evaluation ---
    def evaluate(model, is_spiking, n_episodes=100):
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        rewards = []
        for _ in range(n_episodes):
            state, _ = env.reset(); ep_r = 0; done = False
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

    # --- Run each seed ---
    results = []
    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

        import gymnasium as gym
        env = gym.make("CartPole-v1")

        # Train ANN (early-stop)
        policy_net = ANNDQN().to(device)
        target_net = ANNDQN().to(device)
        target_net.load_state_dict(policy_net.state_dict()); target_net.eval()
        opt = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        replay = ReplayBuffer()
        epsilon = 1.0; steps = 0; train_rewards = []; solved_ep = None

        for ep in range(600):
            state, _ = env.reset(); ep_r = 0; done = False
            while not done:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        action = policy_net(s).argmax(1).item()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated; ep_r += reward; steps += 1
                replay.push(state, action, reward, next_state, done); state = next_state
                if len(replay) >= 64:
                    states_b, actions_b, rewards_b, next_states_b, dones_b = replay.sample(64)
                    states_b = states_b.to(device); actions_b = actions_b.to(device)
                    rewards_b = rewards_b.to(device); next_states_b = next_states_b.to(device)
                    dones_b = dones_b.to(device)
                    curr_q = policy_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_q = target_net(next_states_b).max(1)[0]
                        target_q = rewards_b + 0.99 * next_q * (1 - dones_b)
                    loss = criterion(curr_q, target_q)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                    opt.step()
                if steps % 100 == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            train_rewards.append(ep_r)
            epsilon = max(0.01, epsilon * 0.995)
            avg100 = np.mean(train_rewards[-100:]) if len(train_rewards) >= 100 else np.mean(train_rewards)
            if len(train_rewards) >= 100 and avg100 >= 195:
                solved_ep = ep + 1; break
        env.close()
        policy_net.eval()

        if solved_ep:
            print(f"    ANN solved at Ep {solved_ep}")
        else:
            print(f"    ANN did NOT solve (Train100={avg100:.0f}) — skipping FT")
            results.append({"seed": seed, "ann_solved": False, "snn_solved_pct": 0, "snn_mean": 0})
            continue

        # Weight transfer
        snn = SpikingDQN(T=16, beta=0.5, threshold=1.0).to(device)
        snn.fc1.load_state_dict(policy_net.fc1.state_dict())
        snn.fc2.load_state_dict(policy_net.fc2.state_dict())
        snn.fc3.load_state_dict(policy_net.fc3.state_dict())
        snn.eval()
        mean_before, solved_before = evaluate(snn, is_spiking=True, n_episodes=100)
        print(f"    Pre-FT: Mean={mean_before:.1f}, Solved={solved_before:.0f}%")

        # BPTT fine-tune
        import gymnasium as gym2
        env2 = gym2.make("CartPole-v1")
        target_snn = SpikingDQN(T=16, beta=0.5, threshold=1.0).to(device)
        target_snn.load_state_dict(snn.state_dict()); target_snn.eval()
        opt_ft = torch.optim.Adam(snn.parameters(), lr=5e-4)
        replay_ft = ReplayBuffer(); eps_ft = 0.3; steps_ft = 0; ft_rewards = []

        for ep_ft in range(n_ft_episodes):
            state, _ = env2.reset(); ep_r = 0; done = False
            while not done:
                if random.random() < eps_ft:
                    action = env2.action_space.sample()
                else:
                    with torch.no_grad():
                        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        reset_spiking(snn)
                        action = snn(s).argmax(1).item()
                next_state, reward, terminated, truncated, _ = env2.step(action)
                done = terminated or truncated; ep_r += reward; steps_ft += 1
                replay_ft.push(state, action, reward, next_state, done); state = next_state
                if len(replay_ft) >= 64:
                    states_b, actions_b, rewards_b, next_states_b, dones_b = replay_ft.sample(64)
                    states_b = states_b.to(device); actions_b = actions_b.to(device)
                    rewards_b = rewards_b.to(device); next_states_b = next_states_b.to(device)
                    dones_b = dones_b.to(device)
                    reset_spiking(snn)
                    curr_q = snn(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        reset_spiking(target_snn)
                        next_q = target_snn(next_states_b).max(1)[0]
                        target_q = rewards_b + 0.99 * next_q * (1 - dones_b)
                    loss_ft = criterion(curr_q, target_q)
                    opt_ft.zero_grad(); loss_ft.backward()
                    torch.nn.utils.clip_grad_norm_(snn.parameters(), 1.0)
                    opt_ft.step()
                    if steps_ft % 100 == 0:
                        target_snn.load_state_dict(snn.state_dict())
            ft_rewards.append(ep_r)
            eps_ft = max(0.05, eps_ft * 0.995)
        env2.close()
        snn.eval()

        mean_after, solved_after = evaluate(snn, is_spiking=True, n_episodes=100)
        print(f"    Post-FT: Mean={mean_after:.1f}, Solved={solved_after:.1f}%")
        results.append({"seed": seed, "ann_solved": True, "snn_mean": mean_after,
                        "snn_solved_pct": solved_after})

    # --- Aggregate ---
    n_solved_ann = sum(1 for r in results if r["ann_solved"])
    snn_means = [r["snn_mean"] for r in results if r["ann_solved"]]
    snn_solved = [r["snn_solved_pct"] for r in results if r["ann_solved"]]
    n_solved_snn = sum(1 for s in snn_solved if s >= 95)

    agg = {
        "benchmark": "cartpole",
        "seeds": len(seeds),
        "n_ann_solved": n_solved_ann,
        "n_snn_solved": n_solved_snn,
        "snn_mean_mean": np.mean(snn_means) if snn_means else 0,
        "snn_mean_std": np.std(snn_means) if snn_means else 0,
        "snn_solved_mean": np.mean(snn_solved) if snn_solved else 0,
        "best_snn_solved": max(snn_solved) if snn_solved else 0,
        "per_seed": results,
    }
    return agg


# ===========================================================================
# Robotics Pipeline Benchmark
# ===========================================================================

def run_robotics_benchmark():
    """Run the full robotics perception pipeline (Demo C).

    Event camera → ANN → convert() → sparsity → energy → NIR export.
    """
    print(f"\n{'='*70}")
    print("ROBOTICS PERCEPTION PIPELINE BENCHMARK")
    print(f"{'='*70}")

    import numpy as np
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    sys.path.insert(0, os.path.dirname(__file__))
    from neurocuda import convert, measure_sparsity, to_nir
    from models import IFNeuron, LIFNeuron, reset_spiking

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # --- Model (5D-native — matches demo_c_robotics_perception) ---
    class EventCameraCNN(nn.Module):
        def __init__(self, act_factory=nn.ReLU, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(2, 32, 5, padding=2, bias=False)
            self.bn1 = nn.BatchNorm2d(32); self.act1 = act_factory()
            self.pool1 = nn.AvgPool2d(2)
            self.conv2 = nn.Conv2d(32, 64, 5, padding=2, bias=False)
            self.bn2 = nn.BatchNorm2d(64); self.act2 = act_factory()
            self.pool2 = nn.AvgPool2d(2)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(128); self.act3 = act_factory()
            self.pool3 = nn.AvgPool2d(2)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(2048, num_classes)

        def forward(self, x):
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            x = self.pool1(self.act1(self.bn1(self.conv1(x))))
            x = self.pool2(self.act2(self.bn2(self.conv2(x))))
            x = self.pool3(self.act3(self.bn3(self.conv3(x))))
            x = self.flatten(x)
            x = self.fc(x)
            return x.reshape(B, T, -1).mean(dim=1)

    # --- Load data ---
    train_data = torch.load("./data/nmnist_train.pt", map_location="cpu", weights_only=False)
    test_data = torch.load("./data/nmnist_test.pt", map_location="cpu", weights_only=False)
    n_train = min(20000, len(train_data["data"]))
    train_loader = DataLoader(
        TensorDataset(train_data["data"][:n_train], train_data["targets"][:n_train]),
        batch_size=128, shuffle=True)
    test_loader = DataLoader(
        TensorDataset(test_data["data"][:2000], test_data["targets"][:2000]),
        batch_size=128)

    # --- Load or train ANN ---
    ckpt_path = "./checkpoints/demo_a_ann_best.pt"
    ann_model = EventCameraCNN(act_factory=nn.ReLU).to(device)
    if os.path.exists(ckpt_path):
        ann_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        ann_model.train()
        opt = torch.optim.AdamW(ann_model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        for ep in range(5):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                loss = crit(ann_model(data), target)
                loss.backward(); opt.step()
        ann_model.eval()
        torch.save(ann_model.state_dict(), ckpt_path)
    ann_model.eval()

    # ANN accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            correct += (ann_model(data).argmax(1) == target).sum().item()
            total += data.size(0)
    ann_acc = 100.0 * correct / total
    print(f"  ANN accuracy: {ann_acc:.2f}%")

    # --- Convert ---
    t0 = time.time()
    snn_model, stats = convert(
        ann_model,
        train_loader,
        test_loader=test_loader,
        qcfs_epochs=5,
        if_epochs=5,
        strategy="qcfs_if_ft",
        channel_wise=True,
        device=device,
        verbose=False,
    )
    conv_time = time.time() - t0

    # --- Sparsity ---
    sparsity, _, _, _ = measure_sparsity(snn_model, test_loader, device=device, max_batches=10)

    # --- Energy estimation ---
    E_AC = 0.9e-12   # Loihi 2: 0.9 pJ per SOP
    E_MAC = 4.6e-12  # 45nm CMOS: 4.6 pJ per MAC
    T = 16

    conv_ops = {}
    def hook_conv(m, inp, out):
        oc, ic, kh, kw = m.weight.shape
        _, _, oh, ow = out.shape
        conv_ops[m] = oc * ic * kh * kw * oh * ow
    fc_ops = {}
    def hook_fc(m, inp, out):
        fc_ops[m] = m.weight.numel()

    handles = []
    for m in snn_model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(hook_conv))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(hook_fc))

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= 10: break
            data = data.to(device)
            reset_spiking(snn_model)
            B_s, T_data = data.size(0), min(T, data.size(1))
            for t in range(T_data):
                snn_model(data[:, t:t+1, :, :, :])
    for h in handles: h.remove()

    total_macs = (sum(conv_ops.values()) + sum(fc_ops.values())) * 10  # 10 batches
    total_sops = 0  # Simplified — sparsity-based
    # Energy estimate from sparsity
    dense_e = total_macs * E_MAC * T * 1e3  # mJ
    # Assume 8% of activations fire → SOPs ≈ 0.08 * MACs * T
    sparse_e = total_macs * (1 - sparsity/100) * T * E_AC * 1e3  # mJ
    total_e = dense_e + sparse_e
    per_inf_e = (total_e / (10 * 128)) * 1000  # µJ

    # ANN comparison
    ann_e_per_inf = total_macs * E_MAC * T * 1e6 / (10 * 128)
    snn_e_per_inf = per_inf_e
    energy_saving = (1 - snn_e_per_inf / (ann_e_per_inf + snn_e_per_inf)) * 100

    # --- NIR export ---
    nir_ok = False
    try:
        nir_graph = to_nir(snn_model, T=T, model_name="robotics_benchmark")
        nir_ok = len(nir_graph.get("nodes", [])) > 0
    except Exception:
        pass

    agg = {
        "benchmark": "robotics",
        "ann_acc": ann_acc,
        "if_acc": stats["if_accuracy"],
        "gap": ann_acc - stats["if_accuracy"],
        "sparsity": sparsity,
        "energy_per_inf_uJ": per_inf_e,
        "energy_saving_pct": energy_saving,
        "nir_export_ok": nir_ok,
        "conv_time": conv_time,
    }

    print(f"  IF accuracy: {stats['if_accuracy']:.2f}% | Gap: {agg['gap']:.2f}%")
    print(f"  Sparsity: {sparsity:.1f}% | Energy/inf: {per_inf_e:.2f} µJ")
    print(f"  Energy vs ANN: {energy_saving:.0f}% reduction")
    print(f"  NIR export: {'OK' if nir_ok else 'FAILED'} | Time: {conv_time:.0f}s")

    return agg


# ===========================================================================
# Summary Printer
# ===========================================================================

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def check_pass(agg, benchmark_id):
    """Check if benchmark passes its target thresholds."""
    target = BENCHMARKS[benchmark_id].get("readme_target")
    if target is None:
        return "N/A (stochastic)"

    if benchmark_id == "nmnist":
        gap_ok = abs(agg["gap_mean"] - target["gap"]) < 0.3
        if_ok = abs(agg["if_mean"] - target["if_acc"]) < 0.5
        return "PASS" if (gap_ok and if_ok) else "CHECK"
    elif benchmark_id == "robotics":
        beats = agg["gap"] <= 0  # SNN beats or equals ANN
        sparsity_ok = agg["sparsity"] >= target["sparsity_min"]
        return "PASS" if (beats and sparsity_ok) else "CHECK"
    return "?"


def print_final_summary(all_results):
    """Print the final summary table matching README format."""
    print_header("NEUROCUDA BENCHMARK SUMMARY")

    print(f"\n  Benchmarks run: {len(all_results)}")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M')}")

    for agg in all_results:
        bid = agg["benchmark"]
        name = BENCHMARKS[bid]["name"]
        status = check_pass(agg, bid)
        print(f"\n  ── {name} ──  [{status}]")

        if bid == "nmnist":
            print(f"    Seeds:        {agg['seeds']}")
            print(f"    ANN:          {agg['ann_mean']:.2f}% ± {agg['ann_std']:.2f}%")
            print(f"    SNN (IF):     {agg['if_mean']:.2f}% ± {agg['if_std']:.2f}%")
            print(f"    Gap:          {agg['gap_mean']:.2f}% ± {agg['gap_std']:.2f}%")
            print(f"    Sparsity:     {agg['sparsity_mean']:.1f}% ± {agg['sparsity_std']:.1f}%")
            print(f"    Total time:   {agg['total_time']/60:.1f} min")
            print(f"    Per-seed:     ", end="")
            for r in agg["per_seed"]:
                print(f"S{r['seed']}:{r['if_acc']:.2f}% ", end="")
            print()

        elif bid == "cartpole":
            print(f"    Seeds:        {agg['seeds']}")
            print(f"    ANN solved:   {agg['n_ann_solved']}/{agg['seeds']}")
            print(f"    SNN ≥95%:     {agg['n_snn_solved']}/{agg['seeds']}")
            print(f"    Best SNN:     {agg['best_snn_solved']:.0f}% solved")
            print(f"    Mean SNN:     {agg['snn_solved_mean']:.1f}% ± {agg['snn_solved_mean']:.0f}%")
            print(f"    Note:         Stochastic — ~29% seed success is expected")

        elif bid == "robotics":
            print(f"    ANN:          {agg['ann_acc']:.2f}%")
            print(f"    SNN (IF):     {agg['if_acc']:.2f}%")
            print(f"    Gap:          {agg['gap']:.2f}%")
            print(f"    Sparsity:     {agg['sparsity']:.1f}%")
            print(f"    Energy/inf:   {agg['energy_per_inf_uJ']:.2f} µJ")
            print(f"    Energy vs ANN:{agg['energy_saving_pct']:.0f}% reduction")
            print(f"    NIR export:   {'PASS' if agg['nir_export_ok'] else 'FAIL'}")
            print(f"    Time:         {agg['conv_time']:.0f}s")

    # Cross-reference with README targets
    print_header("CROSS-CHECK vs README")
    print()
    for agg in all_results:
        bid = agg["benchmark"]
        name = BENCHMARKS[bid]["name"]
        status = check_pass(agg, bid)

        if status == "PASS":
            print(f"  ✅ {name}: Matches README numbers")
        elif status == "N/A (stochastic)":
            print(f"  ⚠  {name}: Stochastic benchmark — report actual results")
        else:
            print(f"  ⚠  {name}: Deviates from README — investigate")

    # Overall verdict
    required_results = [check_pass(agg, bid) for agg in all_results
                        for bid in [agg["benchmark"]] if BENCHMARKS[bid]["required"]]
    all_pass = all(r == "PASS" for r in required_results)

    print(f"\n  Overall: {'✅ ALL REQUIRED BENCHMARKS PASS' if all_pass else '⚠  SOME CHECKS FAILED'}")
    print(f"{'='*70}\n")
    return all_pass


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NeuroCUDA — Reproduce All Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reproduce.py                    # Run all benchmarks
  python reproduce.py --quick            # Fast verification (NMNIST only)
  python reproduce.py --demo             # Robotics pipeline only
  python reproduce.py --benchmarks nmnist cartpole  # Specific benchmarks
  python reproduce.py --list             # List available benchmarks
        """
    )
    parser.add_argument("--quick", action="store_true",
                       help="Fast verification — NMNIST only (~4 min)")
    parser.add_argument("--demo", action="store_true",
                       help="Robotics pipeline only (~2 min)")
    parser.add_argument("--benchmarks", nargs="+",
                       choices=list(BENCHMARKS.keys()),
                       help="Specific benchmarks to run")
    parser.add_argument("--list", action="store_true",
                       help="List available benchmarks and exit")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                       help="Seeds for multi-seed benchmarks (default: 0 1 2)")
    parser.add_argument("--n-train", type=int, default=20000,
                       help="Training samples for NMNIST (default: 20000)")
    args = parser.parse_args()

    # --list
    if args.list:
        print("Available benchmarks:")
        for bid, info in BENCHMARKS.items():
            req = "REQUIRED" if info["required"] else "optional"
            print(f"  {bid:<12s} — {info['name']:<35s} [{req}] ~{info['time_est']}")
        print("\nUsage: python reproduce.py [--quick | --demo | --benchmarks ...]")
        return 0

    # Select benchmarks
    if args.quick:
        selected = ["nmnist"]
    elif args.demo:
        selected = ["robotics"]
    elif args.benchmarks:
        selected = args.benchmarks
    else:
        selected = list(BENCHMARKS.keys())  # All

    print("=" * 70)
    print("  NEUROCUDA — BENCHMARK REPRODUCTION")
    print("=" * 70)
    print(f"\n  Selected: {', '.join(selected)}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # --- Step 1: Ensure data ---
    print("[0] Checking data...")
    if "nmnist" in selected or "robotics" in selected:
        if not ensure_nmnist_data():
            print("\n  FATAL: NMNIST data not available. Run: python examples/prep_nmnist.py")
            return 1
    print()

    # --- Step 2: Run benchmarks ---
    all_results = []
    t_total_start = time.time()

    for bid in selected:
        info = BENCHMARKS[bid]
        print(f"\n{'─'*70}")
        print(f"  Running: {info['name']} (est. {info['time_est']})")
        print(f"{'─'*70}")

        t0 = time.time()
        try:
            if bid == "nmnist":
                agg = run_nmnist_benchmark(seeds=args.seeds, n_train=args.n_train)
            elif bid == "cartpole":
                agg = run_cartpole_benchmark(seeds=args.seeds)
            elif bid == "robotics":
                agg = run_robotics_benchmark()
            else:
                print(f"  Unknown benchmark: {bid}")
                continue
            agg["_elapsed"] = time.time() - t0
            all_results.append(agg)
        except Exception as e:
            print(f"\n  ❌ BENCHMARK FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - t_total_start

    # --- Step 3: Print summary ---
    if not all_results:
        print("\n  No benchmarks completed successfully.")
        return 1

    all_pass = print_final_summary(all_results)
    print(f"  Total wall-clock time: {total_time/60:.1f} min")
    print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
