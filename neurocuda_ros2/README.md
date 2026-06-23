# NeuroCUDA ROS2 — Spiking Neural Network Brain for Robots

Put an SNN brain on any ROS2 robot. One command.

```
Camera → SNN → Detections → Foxglove Dashboard
         ↓
    Benchmark (latency, energy, sparsity)
```

---

## What This Does

Takes a trained spiking neural network and runs it on a real or simulated robot:

| Input | Processing | Output |
|-------|-----------|--------|
| Camera image | SNN inference (1.4ms) | What the robot sees |
| Event camera | Spike-driven processing | Detection + confidence |
| Rosbag replay | CPU / GPU / Loihi 2 | Benchmark metrics |

**Key numbers (MLP-MNIST, CPU):** 1.4ms latency | 80% spike sparsity | 91% less energy than dense ANN

---

## Quick Start (Docker — 3 steps)

```bash
# 1. Clone
git clone https://github.com/Krishnav1/neurocuda.git
cd neurocuda

# 2. Start (builds everything automatically)
docker compose up -d

# 3. Enter and run
docker compose exec ros2 bash
source /opt/ros/jazzy/setup.bash
source /neurocuda_ws/install/setup.bash

# Test: SNN inference with simulated camera
ros2 launch neurocuda_ros2 dashboard.launch.py
```

**Open Foxglove Studio** → `ws://localhost:8765` → see live SNN brain activity.

---

## What You Can Do

### 1. Run SNN on Camera Feed
```bash
# Start SNN node + camera simulator
ros2 launch neurocuda_ros2 infer.launch.py model:=neurocuda/mlp-mnist-snn

# Check detections
ros2 topic echo /snn/detections
```

### 2. Gazebo Robot Simulation
```bash
# OGRE2 headless camera world (works without GPU)
ros2 launch neurocuda_ros2 demo_gazebo_headless.launch.py

# Or from CLI:
gz sim -s -r --headless-rendering worlds/ogre2_camera_28.sdf
```

### 3. Benchmark Performance
```bash
# Measure latency, throughput, energy
ros2 launch neurocuda_ros2 benchmark.launch.py

# Watch results
ros2 topic echo /snn/benchmark_summary
```

### 4. Record & Replay (Reproducible Testing)
```bash
# Record a session
ros2 launch neurocuda_ros2 record.launch.py bag_name:=my_test

# Replay + benchmark simultaneously
ros2 launch neurocuda_ros2 benchmark_replay.launch.py bag_file:=my_test

# Same data → same results every time
```

### 5. Foxglove Live Dashboard
```bash
ros2 launch neurocuda_ros2 dashboard.launch.py
# Open Foxglove Studio → ws://localhost:8765
```

### 6. NeuroBench Report
```bash
ros2 run neurocuda_ros2 neurobench --ros-args -p model:=neurocuda/mlp-mnist-snn
# Report saved to /neurocuda_ws/reports/
```

### 7. Multi-Backend Comparison
```bash
ros2 run neurocuda_ros2 multibackend --ros-args -p num_samples:=100
# Compares CPU vs Loihi 2 simulator energy
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    NeuroCUDA ROS2                        │
│                                                         │
│  Input Sources:                                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐             │
│  │ Camera   │  │ Event    │  │ Rosbag    │             │
│  │ (Image)  │  │ Camera   │  │ (.mcap)   │             │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘             │
│       │             │              │                    │
│       └──────────┬──┘──────────────┘                    │
│                  ▼                                      │
│         ┌────────────────┐                              │
│         │  SNN Inference │  Lifecycle Node              │
│         │  (IF/LIF)      │  Configure → Activate        │
│         └───────┬────────┘                              │
│                 ▼                                       │
│  ┌──────────────┼──────────────┐                        │
│  ▼              ▼              ▼                        │
│  Detections   Spikes    Sparsity/Status                 │
│                                                         │
│  Output Destinations:                                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐             │
│  │ Foxglove │  │ Rosbag   │  │ NeuroBench│             │
│  │ Dashboard│  │ Record   │  │ Report    │             │
│  └──────────┘  └──────────┘  └───────────┘             │
│                                                         │
│  Backends: GPU (CUDA) | CPU | Loihi 2 (simulator)       │
└─────────────────────────────────────────────────────────┘
```

---

## Topics Reference

| Topic | Type | What It Is |
|-------|------|------------|
| `/camera/image` | `sensor_msgs/Image` | Camera input (any resolution, any encoding) |
| `/dvs/events` | `event_camera_msgs/EventArray` | Event camera input (sparse) |
| `/snn/detections` | `neurocuda_msgs/SnnDetection` | What the SNN sees: class, confidence, top-k labels |
| `/snn/spikes` | `neurocuda_msgs/SnnSpikeEvent` | Per-layer spike counts and sparsity |
| `/snn/sparsity` | `std_msgs/Float32` | Overall spike sparsity % |
| `/snn/status` | `neurocuda_msgs/SnnStatus` | Model info: accuracy, params, device |
| `/snn/benchmark` | `neurocuda_msgs/SnnStatus` | Per-inference latency + energy |
| `/snn/benchmark_summary` | `std_msgs/String` | Periodic benchmark report |
| `/snn/neurobench_report` | `std_msgs/String` | NeuroBench v2.0 standard report (JSON) |

---

## Launch Files

| File | What It Does |
|------|-------------|
| `infer.launch.py` | SNN inference + lifecycle manager |
| `dashboard.launch.py` | SNN + Foxglove bridge + camera sim |
| `benchmark.launch.py` | SNN + benchmark node + camera sim |
| `benchmark_replay.launch.py` | Rosbag replay → SNN + benchmark |
| `record.launch.py` | Record all topics to rosbag |
| `replay.launch.py` | Play rosbag into SNN pipeline |
| `demo_synthetic.launch.py` | Hardware-free test (camera_sim + SNN) |
| `demo_gazebo.launch.py` | Gazebo + TurtleBot4 + SNN |
| `demo_gazebo_headless.launch.py` | Headless Gazebo + SNN (no GPU) |

---

## Real Numbers (Docker, CPU, MLP-MNIST-SNN)

```
Model:      MLP-MNIST-SNN (269K params, 2 IF neurons)
Accuracy:   97.4% (published), 86.7% (ROS2 pipeline, trained weights)
Dataset:    MNIST (handwritten digits 0-9)

Latency:    avg=1.4ms  p50=0.9ms  p95=1.9ms
Throughput: 4.8 inferences/sec
Sparsity:   80.5% (only 100 of 512 neurons fire)
Memory:     1.0 MB (float32)

Energy (modeled, Loihi 2 constants):
  SNN sparse:   0.73 µJ/inference
  ANN dense:    215 µJ/inference
  Reduction:    296× (99.7%)

Honest notes:
  - Energy numbers are MODELED from published constants, NOT measured on silicon
  - Loihi 2 = simulator only (8-bit weights, 24-bit membrane)
  - GPU backend unavailable in Docker (CPU-only)
  - Accuracy gap (86.7% vs 97.4%) from ANN→SNN weight conversion without QCFS calibration
  - For published accuracy, use the HuggingFace weights or run neurocuda.convert()
```

---

## Install Without Docker (Ubuntu 24.04)

```bash
# 1. Install ROS2 Jazzy
#    https://docs.ros.org/en/jazzy/Installation.html

# 2. Install Gazebo Harmonic
sudo apt install gz-harmonic ros-jazzy-ros-gz-bridge

# 3. Install Foxglove bridge
sudo apt install ros-jazzy-foxglove-bridge

# 4. Install NeuroCUDA
pip install neurocuda --break-system-packages

# 5. Build workspace
mkdir -p ~/neurocuda_ws/src
cp -r neurocuda_msgs ~/neurocuda_ws/src/
cp -r neurocuda_ros2 ~/neurocuda_ws/src/
cd ~/neurocuda_ws
colcon build --packages-select neurocuda_msgs neurocuda_ros2
source install/setup.bash

# 6. Run
ros2 launch neurocuda_ros2 dashboard.launch.py
```

---

## Hardware Targets

| Backend | What It Is | How To Use |
|---------|-----------|------------|
| **CPU** | PyTorch on x86 | `-p device:=cpu` (default) |
| **GPU** | CUDA PyTorch | `-p device:=cuda` |
| **Loihi 2** | Simulator (8-bit quantized) | `-p hardware_target:=loihi` |
| **FPGA** | SC-NeuroCore | Coming soon |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: rclpy` | Source ROS2: `source /opt/ros/jazzy/setup.bash` |
| `No transition matching activate` | Node state issue. Check with: `ros2 service call /snn_inference/get_state` |
| Gazebo camera shows 0 publishers | Use OGRE2 render engine + `--headless-rendering` flag |
| Foxglove can't connect | Check port: `docker ps` should show `8765→8765` |
| Model weights not found | Train weights or download from HuggingFace hub |
| NumPy 1.x/2.x error in cv_bridge | Ignored — we use numpy-only fallback. Does not affect SNN. |

---

## License

MIT — see [LICENSE](../LICENSE)
