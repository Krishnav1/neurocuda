# NeuroCUDA ROS2

**Spiking Neural Network inference and control for ROS2 robots.**

One command to add SNN perception and control to any ROS2 system.

## Installation

```bash
# 1. Install NeuroCUDA
pip install neurocuda

# 2. Install ROS2 package (from source)
cd neurocuda_ros2
pip install -e .

# Or copy to ROS2 workspace
cp -r neurocuda_ros2 ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select neurocuda_ros2
```

## Quick Start

```bash
# Infer with event camera SNN
ros2 run neurocuda_ros2 snn_infer --ros-args -p model:=cnn-nmnist-snn

# Control robot with spiking DQN
ros2 run neurocuda_ros2 snn_control --ros-args -p model:=dqn-cartpole-snn

# Monitor spike activity
ros2 run neurocuda_ros2 spike_viz --ros-args -p model:=cnn-nmnist-snn
```

## Launch Files

```bash
# SNN inference pipeline
ros2 launch neurocuda_ros2 infer.launch.py model:=cnn-nmnist-snn

# SNN control pipeline
ros2 launch neurocuda_ros2 control.launch.py model:=dqn-cartpole-snn

# Full NMNIST demo
ros2 launch neurocuda_ros2 demo_nmnist.launch.py
```

## Topics

### Inference Node
| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/camera/image` | `sensor_msgs/Image` | Sub | Regular camera input |
| `/dvs/events` | `event_camera_msgs/EventArray` | Sub | Event camera input |
| `/snn/detections` | `neurocuda_msgs/SnnDetection` | Pub | Structured detection (class, confidence, top-k, sparsity) |
| `/snn/spikes` | `neurocuda_msgs/SnnSpikeEvent` | Pub | Per-layer spike activity |
| `/snn/sparsity` | `std_msgs/Float32` | Pub | Spike sparsity % (lightweight) |
| `/snn/status` | `neurocuda_msgs/SnnStatus` | Pub | Model status and metrics |

### Control Node
| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/robot/state` | `std_msgs/Float32MultiArray` | Sub | Robot state |
| `/odom` | `nav_msgs/Odometry` | Sub | Odometry state |
| `/cmd_vel` | `geometry_msgs/Twist` | Pub | Velocity commands |
| `/snn/action` | `neurocuda_msgs/SnnStatus` | Pub | Action + model status |
| `/snn/q_values` | `std_msgs/Float32MultiArray` | Pub | Q-values |

## Test Without ROS2

```bash
# Run synthetic demo (no ROS2 required)
python scripts/demo_synthetic.py
python scripts/demo_synthetic.py --model robotics-perception-snn --pattern moving_bar
```

## Supported Hardware

| Hardware | Backend | Command |
|----------|---------|---------|
| GPU (CUDA) | PyTorch | `--ros-args -p device:=cuda` |
| CPU | PyTorch | `--ros-args -p device:=cpu` |
| Loihi 2 | Lava | Coming soon |
| FPGA | SC-NeuroCore | Coming soon |

## License

MIT — see [LICENSE](../LICENSE)
