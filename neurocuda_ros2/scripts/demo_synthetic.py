#!/usr/bin/env python3
"""
NeuroCUDA ROS2 Demo — Synthetic Event Camera → SNN → Detection

Tests the full pipeline WITHOUT requiring ROS2 or a real event camera.
Uses synthetic event data generated on the fly.

Usage:
    python scripts/demo_synthetic.py
    python scripts/demo_synthetic.py --model cnn-nmnist-snn --steps 100

What it demonstrates:
    1. Load SNN model from NeuroCUDA hub
    2. Generate synthetic event camera data
    3. Run continuous SNN inference with stateful IF neurons
    4. Measure and print spike statistics
    5. Show detection results
"""

import sys, os, time, argparse
import numpy as np
import torch

# Add repo root and neurocuda_ros2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import model_loader directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "model_loader",
    os.path.join(os.path.dirname(__file__), "..", "neurocuda_ros2", "model_loader.py")
)
model_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_loader)
ModelLoader = model_loader.ModelLoader
events_to_tensor = model_loader.events_to_tensor
detection_to_msg = model_loader.detection_to_msg


def generate_synthetic_events(T=16, H=34, W=34, pattern="random"):
    """Generate synthetic event data for testing.

    Args:
        T: number of temporal bins
        H, W: frame dimensions
        pattern: "random", "moving_bar", "circle"

    Returns:
        torch.Tensor of shape (1, T, 2, H, W)
    """
    events = torch.zeros(1, T, 2, H, W)

    if pattern == "random":
        # Random sparse events
        num_events = 500
        for _ in range(num_events):
            t = np.random.randint(0, T)
            p = np.random.randint(0, 2)
            x = np.random.randint(0, W)
            y = np.random.randint(0, H)
            events[0, t, p, y, x] += 1.0

    elif pattern == "moving_bar":
        # Vertical bar moving left to right
        for t in range(T):
            x_pos = int(t * W / T)
            p = 1 if t % 2 == 0 else 0  # alternating polarity
            events[0, t, p, :, max(0, x_pos-2):min(W, x_pos+2)] = 1.0

    elif pattern == "circle":
        # Expanding circle
        center = (H // 2, W // 2)
        for t in range(T):
            radius = int(5 + t * min(H, W) / T / 3)
            p = t % 2
            for y in range(max(0, center[0]-radius-1), min(H, center[0]+radius+1)):
                for x in range(max(0, center[1]-radius-1), min(W, center[1]+radius+1)):
                    dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)
                    if abs(dist - radius) < 1.5:
                        events[0, t, p, y, x] = 1.0

    return torch.clamp(events, 0, 5.0)


def main():
    parser = argparse.ArgumentParser(description="NeuroCUDA ROS2 Synthetic Demo")
    parser.add_argument("--model", default="neurocuda/cnn-nmnist-snn",
                       help="SNN model to use")
    parser.add_argument("--pattern", default="random",
                       choices=["random", "moving_bar", "circle"],
                       help="Synthetic event pattern")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--device", default="auto",
                       help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("  NeuroCUDA ROS2 — Synthetic Demo")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Pattern: {args.pattern}")
    print(f"  Steps: {args.steps}")
    print()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"  Device: {device}")

    # Load model
    print(f"\n[1] Loading SNN model...")
    t0 = time.time()
    loader = ModelLoader(args.model, device=device)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Architecture: {loader.num_params:,} params")
    print(f"  Spiking neurons: {loader.if_count} IF + {loader.lif_count} LIF")
    print(f"  Accuracy: {loader.accuracy}%")

    # Run inference loop
    print(f"\n[2] Running {args.steps} inference steps...")
    print(f"  Pattern: {args.pattern}")
    total_time = 0
    total_spikes = 0
    detection_counts = {}

    loader.reset_state()

    for step in range(args.steps):
        # Generate synthetic events
        events = generate_synthetic_events(T=16, pattern=args.pattern)
        events = events.to(device)

        # Run SNN inference
        t_step = time.time()
        output, spike_stats = loader.infer_5d(events)
        step_time = (time.time() - t_step) * 1000  # ms

        total_time += step_time
        total_spikes += spike_stats["total_spikes"]

        # Detection
        result = detection_to_msg(output)
        detection_counts[result["class_name"]] = \
            detection_counts.get(result["class_name"], 0) + 1

        # Progress
        if (step + 1) % 10 == 0:
            avg_time = total_time / (step + 1)
            avg_spikes = total_spikes / (step + 1)
            print(f"  Step {step+1:3d}/{args.steps} | "
                  f"{step_time:5.1f}ms | "
                  f"Spikes: {spike_stats['total_spikes']:6,}/{spike_stats['total_activations']:7,} "
                  f"({spike_stats['sparsity']:.1f}% sparse) | "
                  f"Detected: {result['class_name']} ({result['confidence']:.2f})")

    # Summary
    print(f"\n[3] Summary")
    print(f"  Total time: {total_time/1000:.1f}s ({total_time/args.steps:.1f}ms avg)")
    print(f"  Avg sparsity: {100*(1 - total_spikes/(args.steps*loader.num_params)):.1f}%")
    print(f"  Detections: {dict(sorted(detection_counts.items()))}")

    print(f"\n✅ ROS2-compatible pipeline working!")
    print(f"   Real ROS2 usage:")
    print(f"   ros2 run neurocuda_ros2 snn_infer --ros-args -p model:={args.model.split('/')[-1]}")
    print(f"   ros2 launch neurocuda_ros2 demo_nmnist.launch.py model:={args.model.split('/')[-1]}")


if __name__ == "__main__":
    main()
