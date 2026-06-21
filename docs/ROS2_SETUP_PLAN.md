# NeuroCUDA × ROS2 — Complete Setup Plan

## Research Summary

ROS2 Jazzy (latest, June 2026) runs on Ubuntu 24.04. The industry standard for ROS2 + GPU development is Docker with NVIDIA Container Toolkit. This is how NVIDIA's own Isaac ROS ships, and it's the recommended approach by every major robotics project in 2025-2026.

## The Setup (3 Options)

### Option A: Docker (RECOMMENDED) 🥇

Works on your Windows laptop with RTX 5050 GPU. Industry standard. One-time setup.

**Host (Windows):**
```bash
# 1. Install Docker Desktop for Windows
# 2. Install NVIDIA Container Toolkit for Windows
# 3. Verify: docker run --rm --gpus all nvidia/cuda:12.6-base nvidia-smi
```

**Dockerfile (we build):**
```dockerfile
FROM nvidia/cuda:12.6-devel-ubuntu24.04

# ROS2 Jazzy
RUN apt update && apt install -y curl software-properties-common
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" > /etc/apt/sources.list.d/ros2.list
RUN apt update && apt install -y ros-jazzy-ros-base python3-colcon-common-extensions python3-pip

# PyTorch + NeuroCUDA
RUN pip install torch numpy neurocuda

# neurocuda_ros2 package
COPY neurocuda_ros2/ /workspace/src/neurocuda_ros2/
RUN cd /workspace && . /opt/ros/jazzy/setup.sh && colcon build

# Entrypoint
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
RUN echo "source /workspace/install/setup.bash" >> ~/.bashrc
```

**Usage:**
```bash
# Build
docker build -t neurocuda-ros2 -f Dockerfile.ros2 .

# Run with GPU + ROS2
docker run -it --gpus all --network=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  neurocuda-ros2 bash

# Inside container:
ros2 run neurocuda_ros2 snn_infer --ros-args -p model:=cnn-nmnist-snn
ros2 launch neurocuda_ros2 demo_nmnist.launch.py
```

**Pros:** Portable, reproducible, GPU passthrough, industry standard, works on Windows
**Cons:** Docker Desktop overhead, X11 forwarding for RViz

### Option B: Native Ubuntu (Dual-Boot) 🥈

Best performance. Full GPU + display. No Docker overhead.

**Setup:**
1. Install Ubuntu 24.04 alongside Windows (50GB partition)
2. Install NVIDIA drivers: `sudo apt install nvidia-driver-560`
3. Install ROS2 Jazzy: Standard apt install
4. Install PyTorch CUDA: `pip install torch`
5. Install NeuroCUDA: `pip install neurocuda`
6. Copy neurocuda_ros2 to workspace, colcon build

**Pros:** Native GPU, full display, best performance
**Cons:** Requires partition, dual-boot, manual setup

### Option C: WSL2 (If You Already Have It) 🥉

Simplest for quick tests. Limited display support.

The WSL2 Ubuntu 24.04 on your machine works. ROS2 can run without GUI (RViz not needed for SNN inference).

```bash
wsl bash
# Install ROS2 Jazzy (same as Docker steps)
# Install PyTorch + NeuroCUDA
# Build and run
```

**Pros:** No extra setup, already have WSL
**Cons:** No GPU passthrough in WSL, no RViz, not production-grade

## What We Build

### Dockerfile.ros2

```dockerfile
FROM nvidia/cuda:12.6-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

# Install ROS2 Jazzy
RUN apt-get update && apt-get install -y curl gnupg2 lsb-release software-properties-common
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" > /etc/apt/sources.list.d/ros2.list
RUN apt-get update && apt-get install -y ros-jazzy-ros-base python3-colcon-common-extensions python3-pip python3-rosdep2
RUN rosdep init && rosdep update

# Install PyTorch CUDA + NeuroCUDA
RUN pip3 install torch numpy neurocuda

# Workspace setup
RUN mkdir -p /workspace/src
WORKDIR /workspace

# Environment
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

# Default entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
```

### entrypoint.sh

```bash
#!/bin/bash
source /opt/ros/jazzy/setup.bash
exec "$@"
```

### Build & Run

```bash
# Build
docker build -t neurocuda:ros2 -f Dockerfile.ros2 .

# Run (GPU + ROS2 networking)
docker run -it --rm \
  --gpus all \
  --network=host \
  --ipc=host \
  -v $(pwd)/neurocuda_ros2:/workspace/src/neurocuda_ros2 \
  -v $(pwd):/neurocuda \
  neurocuda:ros2 bash

# Inside container:
cd /workspace
colcon build --packages-select neurocuda_ros2
source install/setup.bash

# Run the SNN inference node
ros2 run neurocuda_ros2 snn_infer --ros-args -p model:=cnn-nmnist-snn

# Run the full demo
ros2 launch neurocuda_ros2 demo_nmnist.launch.py

# List topics
ros2 topic list
# /snn/detections
# /snn/sparsity
# /snn/status
# /snn/spike_raster
```

## Verification Checklist

After setup, verify each step:

| Step | Command | Expected Output |
|------|---------|----------------|
| GPU in Docker | `nvidia-smi` | RTX 5050, 8GB |
| ROS2 | `ros2 --version` | Jazzy Jalisco |
| rclpy | `python3 -c "import rclpy"` | OK |
| NeuroCUDA | `python3 -c "import neurocuda"` | 0.2.0 |
| PyTorch CUDA | `python3 -c "import torch; print(torch.cuda.is_available())"` | True |
| Build | `colcon build --packages-select neurocuda_ros2` | Success |
| Run node | `ros2 run neurocuda_ros2 snn_infer` | Node starts, topics created |
| Check topics | `ros2 topic list` | /snn/* topics visible |

## What This Delivers

After setup, anyone can:

```bash
# 1. Clone and build
git clone https://github.com/neurocuda/neurocuda
cd neurocuda
docker build -t neurocuda:ros2 -f Dockerfile.ros2 .
docker run -it --gpus all neurocuda:ros2 bash

# 2. Run SNN on ROS2
ros2 launch neurocuda_ros2 demo_nmnist.launch.py

# 3. Or one-line
ros2 run neurocuda_ros2 snn_infer --ros-args -p model:=cnn-nmnist-snn
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `Dockerfile.ros2` | ROS2 + CUDA + NeuroCUDA image |
| `entrypoint.sh` | Source ROS2 on container start |
| `docker-compose.yml` | Optional: easier multi-container setup |
| `.dockerignore` | Exclude large files from build context |

## Timeline

| Step | Time |
|------|------|
| Create Dockerfile | 30 min |
| Build image | 10 min (download) |
| Copy & build ROS2 package in container | 2 min |
| Run and verify nodes | 5 min |
| **TOTAL** | **< 1 hour** |
