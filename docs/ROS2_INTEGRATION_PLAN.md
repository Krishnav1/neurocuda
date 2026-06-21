# NeuroCUDA × ROS2 — Complete Integration Plan

> Based on deep research across 20+ papers, GitHub repos, ROS2 package index, and market data (June 2026)

---

## 1. WHAT IS ROS2? (Simple Explanation)

ROS2 (Robot Operating System 2) is the **standard software for robots**. It lets different parts of a robot communicate:

```
┌──────────┐    /camera/image    ┌──────────┐    /detections    ┌──────────┐
│  Camera  │ ──────────────────→ │   AI     │ ────────────────→ │  Control │
│  Node    │                     │  Node    │                   │  Node    │
└──────────┘                     └──────────┘                   └──────────┘
```

Every major robotics company, drone manufacturer, and research lab uses ROS2. It's the **universal language of robots.**

---

## 2. WHAT ALREADY EXISTS

### Mature — Production Ready

| Tool | What It Does | Status |
|------|-------------|--------|
| `metavision_driver` | Prophesee event cameras → ROS2 | ✅ v3.0, active April 2026 |
| `event_camera_msgs` | Standard ROS2 message types for events | ✅ Active |
| `event_camera_renderer` | Events → image frames | ✅ Active |
| `event_camera_py` | Fast Python event decoding | ✅ Active |
| `Isaac ROS` (NVIDIA) | CNN/Transformer inference for ROS2 (TensorRT) | ✅ Mature |
| `ros2ai` | CLI tool: natural language → ROS2 commands | ✅ Active |
| `ORICF` | Modular inference framework for ROS2 | ✅ ICRA 2026 |

### Research-Only — Custom, Not Reusable

| Project | What It Does | Why Not Reusable |
|---------|-------------|-----------------|
| **Eldarin** | GPS-denied drone nav with SNN + FPGA | Complete custom system, no package |
| **Purdue DVS-Nav** | Event camera + SNN drone avoidance | Gazebo simulation, custom code |
| **TU Delft SNN Control** | SNN quadcopter control (better than ANN!) | Thesis code, not packaged |
| **NeuEdge** | SNN framework for edge AI | Academic framework, no ROS2 |

### COMPLETELY MISSING

| What Should Exist | Current State |
|-------------------|---------------|
| **ROS2 node that runs SNN inference** | ❌ DOES NOT EXIST |
| **Event camera → SNN pipeline in ROS2** | ❌ DOES NOT EXIST |
| **SNN control node for ROS2** | ❌ DOES NOT EXIST |
| **Standard ROS2 message types for spikes** | ❌ DOES NOT EXIST |
| **Pre-packaged SNN models for ROS2** | ❌ DOES NOT EXIST |

---

## 3. THE OPPORTUNITY

### The Gap

ROS2 has **mature event camera support** and **mature ANN inference** (Isaac ROS, ORICF). But the middle piece — **spiking neural network inference as a reusable ROS2 package** — is completely empty.

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ metavision_      │     │  neurocuda_ros2  │     │  Your Robot      │
│ driver           │ ──→ │  (THIS IS NEW)   │ ──→ │  Controller      │
│ (EXISTS ✅)      │     │  (GAP ❌)        │     │  (EXISTS ✅)     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
   Events topic            SNN Inference            /cmd_vel, /goal
   /dvs/events             /detections              /action
```

### Why NOW

1. **Event cameras are commercially available** — Prophesee Gen4, Sony HVS, Samsung DVS, iniVation
2. **ROS2 event camera drivers are mature** — metavision_driver v3.0, active development
3. **SNN hardware is shipping** — Loihi 2, Speck, Pulsar, SpiNNaker2, Akida
4. **Papers are screaming for this** — every 2025-2026 drone+SNN paper builds custom pipelines
5. **Energy savings proven** — 312× vs GPU for drone workloads
6. **SNNs can outperform ANNs** — TU Delft: SNN beat ANN in real flight tests
7. **NO ONE has built this yet** — we can be first

---

## 4. USE CASES (Who Needs This TODAY)

### Use Case 1: Drone Perception 🥇

**Problem**: Drone needs to see objects with minimal power. GPU drains battery in minutes.
**Solution**: Event camera + SNN on neuromorphic chip. Hours of flight instead of minutes.

```bash
ros2 run neurocuda infer \
  --model neurocuda/cnn-nmnist-snn \
  --event-topic /dvs/events \
  --output-topic /detections \
  --hardware loihi2
```

**Real numbers**: 312× energy reduction. 0.25W vs 78W GPU. Multi-hour drone missions.

### Use Case 2: Robot Control 🥈

**Problem**: Robot needs fast, low-latency control policy.
**Solution**: Spiking DQN running directly on the robot's ROS2 stack.

```bash
ros2 run neurocuda control \
  --model neurocuda/dqn-cartpole-snn \
  --state-topic /robot/state \
  --action-topic /cmd_vel
```

**Real numbers**: TU Delft — SNN beat ANN in quadcopter gate navigation (70.63 vs 59.77 reward, 7.94 vs 6.99 m/s).

### Use Case 3: Always-On Industrial Monitoring 🥉

**Problem**: Factory needs 24/7 anomaly detection without cloud dependency.
**Solution**: Event camera + SNN on edge hardware. Never sends data to cloud.

```bash
ros2 run neurocuda infer \
  --model neurocuda/anomaly-detection-snn \
  --event-topic /dvs/events \
  --output-topic /anomalies \
  --hardware speck
```

**Real numbers**: <1 mW inference. Continuous monitoring for years on battery.

### Use Case 4: Swarm Robotics

**Problem**: Multiple drones need coordinated perception without central server.
**Solution**: Each drone runs its own SNN via NeuroCUDA ROS2 node.

**Real numbers**: U Oklahoma — 15 indoor + 6 outdoor UAVs with SNN control. 90% compute reduction vs traditional.

---

## 5. THE BUILD PLAN

### Phase 1: Core ROS2 Package (Day 1-3)

```
neurocuda_ros2/
├── package.xml                  # ROS2 package metadata
├── setup.py                     # Python package
├── neurocuda_ros2/
│   ├── __init__.py
│   ├── snn_inference_node.py    # Main inference node
│   ├── snn_control_node.py      # Control/policy node
│   └── spike_viz.py             # Spike visualization tool
├── launch/
│   ├── infer.launch.py          # Launch inference pipeline
│   ├── control.launch.py        # Launch control pipeline
│   └── demo_nmnist.launch.py    # Full NMNIST demo
├── config/
│   └── models.yaml              # Model registry for ROS2
└── README.md
```

### Key Node: `snn_inference_node.py`

```python
class SNNInferenceNode(Node):
    def __init__(self):
        # Subscribe to camera/events
        self.create_subscription(Image, '/camera/image', self.image_callback)
        self.create_subscription(EventArray, '/dvs/events', self.event_callback)

        # Load SNN model from NeuroCUDA hub
        self.snn, self.info = nc.hub.load(self.model_name)

        # Publish results
        self.detection_pub = self.create_publisher(DetectionArray, '/detections')
        self.spike_pub = self.create_publisher(SpikeArray, '/spikes')

    def event_callback(self, msg):
        # Convert ROS event message → tensor
        events = event_msg_to_tensor(msg)

        # Run SNN — binary IF spikes, stateful membrane
        with torch.no_grad():
            output = self.snn(events)

        # Publish: detections + spike activity
        self.detection_pub.publish(tensor_to_detection(output))
        self.spike_pub.publish(measure_spikes(self.snn))
```

### Key Node: `snn_control_node.py`

```python
class SNNControlNode(Node):
    def __init__(self):
        self.create_subscription(Odometry, '/robot/state', self.state_callback)
        self.snn, self.info = nc.hub.load(self.model_name)
        self.action_pub = self.create_publisher(Twist, '/cmd_vel')

    def state_callback(self, msg):
        state = odom_to_tensor(msg)
        with torch.no_grad():
            action_values = self.snn(state)  # Q-values from spiking DQN
        action = action_values.argmax()
        self.action_pub.publish(action_to_twist(action))
```

### Phase 2: Demo + Tutorials (Day 4-5)

```bash
# 1. Install
sudo apt install ros-${ROS_DISTRO}-neurocuda

# 2. Launch NMNIST demo (simulated event camera)
ros2 launch neurocuda_ros2 demo_nmnist.launch.py

# 3. Real event camera
ros2 launch neurocuda_ros2 infer.launch.py \
  model:=neurocuda/cnn-nmnist-snn \
  camera:=prophesee_evk4

# 4. RViz2 visualization
ros2 run neurocuda_ros2 spike_viz --ros-args -p model:=cnn-nmnist-snn
```

### Phase 3: Hardware Acceleration (Week 2)

```bash
# Loihi 2
ros2 run neurocuda infer --hardware loihi2

# FPGA (SC-NeuroCore)
ros2 run neurocuda infer --hardware fpga

# CPU fallback
ros2 run neurocuda infer --hardware cpu
```

---

## 6. COMPETITIVE ANALYSIS

### Who Could Build This

| Player | What They Have | Why NeuroCUDA Wins |
|--------|---------------|-------------------|
| **NVIDIA Isaac ROS** | CNN/Transformer inference | GPU-only. No SNN. High power. |
| **Intel Lava** | Loihi-optimized SNNs | No ROS2. Complex. Research-only. |
| **BrainChip MetaTF** | Akida deployment | Vendor-locked. No ROS2. |
| **SynSense Sinabs** | Speck deployment | Vendor-locked. No ROS2. |
| **Eldarin (open source)** | Full drone SNN stack | Custom system, not a reusable package |
| **NeuroCUDA** | **Multi-model, multi-hardware, multi-backend, one command, ROS2-native** | **The only player with all pieces** |

### The Moat

Once NeuroCUDA ROS2 is the standard:
- Every robotics lab uses it → every paper cites it → more models published → more users
- Every neuromorphic chip vendor wants compatibility
- Network effect: the package with the most models and most hardware support wins

---

## 7. WHY THIS BENEFITS NEUROCUDA

### Direct Benefits

| Benefit | How |
|---------|-----|
| **Users** | Every ROS2 robot (~100K+ active developers) can now use NeuroCUDA |
| **Citations** | Every paper using the ROS2 node cites NeuroCUDA |
| **Models** | More users → more models trained → hub grows |
| **Revenue path** | Enterprise support for ROS2 + neuromorphic deployment |
| **Hardware partnerships** | Chip vendors want their hardware supported in NeuroCUDA |
| **Community** | Open source → contributors → faster development |

### Market Numbers

| Market | Size | Our Entry |
|--------|------|-----------|
| ROS2 developers | 100K+ | `ros2 run neurocuda` |
| Drone/UAV market | $12.5B (2026) | Drone perception + control |
| Industrial robotics | $20B+ | Anomaly detection, predictive maintenance |
| Defense robotics | $15B+ | Low-power autonomous systems |
| Neuromorphic software | Fastest growing segment | The deployment layer |

---

## 8. TIMELINE

```
WEEK 1:                     WEEK 2:                     WEEK 3:
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ Core ROS2 pkg   │        │ Hardware        │        │ Release +       │
│ - Inference node│   →    │ acceleration    │   →    │ Tutorials       │
│ - Control node  │        │ - Loihi2        │        │ - Blog post     │
│ - Event camera  │        │ - FPGA stub     │        │ - Video demo    │
│ - Demo launch   │        │ - GPU/CPU       │        │ - Documentation │
└─────────────────┘        └─────────────────┘        └─────────────────┘
```

### Success Metrics

- [ ] `ros2 run neurocuda infer` works with simulated event data
- [ ] `ros2 run neurocuda infer` works with real Prophesee camera
- [ ] `ros2 run neurocuda control` works with CartPole simulation
- [ ] 4 models available as ROS2 parameters
- [ ] RViz2 visualization of spike activity
- [ ] Demo video: event camera → SNN → detection in real time
- [ ] Documentation on ROS Index
- [ ] Package submitted to ROS2 build farm

---

## 9. THE SIMPLE PITCH

**Before NeuroCUDA ROS2:**
"I want my drone to use a spiking neural network."
→ Learn Lava SDK → Convert ANN to SNN → Write custom ROS2 node from scratch → Debug spike messages → Deploy → 3 months later...

**After NeuroCUDA ROS2:**
```bash
ros2 run neurocuda infer --model cnn-nmnist-snn
```
→ Done. 10 seconds.

That's the value. That's what makes people use it.
