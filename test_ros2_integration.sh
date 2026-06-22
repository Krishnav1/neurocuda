#!/bin/bash
# End-to-end ROS2 integration test for NeuroCUDA
set -e

source /opt/ros/jazzy/setup.bash
cd /neurocuda_ws

echo "=========================================="
echo "  STEP 1: Build packages"
echo "=========================================="
colcon build --packages-select neurocuda_msgs neurocuda_ros2
source install/setup.bash

echo ""
echo "=========================================="
echo "  STEP 2: Python message create + populate"
echo "=========================================="
python3 << 'EOF'
from neurocuda_msgs.msg import SnnDetection, SnnSpikeEvent, SnnStatus

# Full SnnDetection
d = SnnDetection()
d.class_id = 3
d.class_name = "cat"
d.confidence = 0.942
d.top_k_labels = ["cat", "dog", "bird"]
d.top_k_scores = [0.942, 0.031, 0.012]
d.sparsity = 87.3
d.total_spikes = 1456
d.total_neurons = 11488
assert d.confidence > 0.9
assert d.class_name == "cat"
assert len(d.top_k_labels) == 3
print(f"PASS SnnDetection: {d.class_name} ({d.confidence:.3f}) spikes={d.total_spikes}/{d.total_neurons}")

# Full SnnSpikeEvent
s = SnnSpikeEvent()
s.layer_name = "act1"
s.neuron_type = "IF"
s.spike_count = 234
s.total_neurons = 784
s.sparsity = 70.2
assert s.layer_name == "act1"
assert s.neuron_type == "IF"
print(f"PASS SnnSpikeEvent: {s.layer_name} ({s.neuron_type}) {s.spike_count}/{s.total_neurons}")

# Full SnnStatus
st = SnnStatus()
st.model_name = "neurocuda/vgg5-cifar10-snn"
st.task = "vision"
st.architecture = "VGG-5 with IF neurons"
st.accuracy = 94.61
st.total_params = 1950000
st.neuron_count = 4096
st.device = "cpu"
st.avg_sparsity = 87.3
st.inference_time_ms = 12.5
assert st.accuracy > 90.0
assert st.device == "cpu"
print(f"PASS SnnStatus: {st.model_name} acc={st.accuracy}% device={st.device}")

print("ALL MESSAGE TESTS PASSED")
EOF

echo ""
echo "=========================================="
echo "  STEP 3: ros2 interface show"
echo "=========================================="
echo "--- SnnDetection ---"
ros2 interface show neurocuda_msgs/msg/SnnDetection

echo ""
echo "--- SnnSpikeEvent ---"
ros2 interface show neurocuda_msgs/msg/SnnSpikeEvent

echo ""
echo "--- SnnStatus ---"
ros2 interface show neurocuda_msgs/msg/SnnStatus

echo ""
echo "=========================================="
echo "  STEP 4: Model loader imports"
echo "=========================================="
python3 << 'EOF'
from neurocuda_ros2.model_loader import ModelLoader, detection_to_msg, spike_stats_to_msg, status_to_msg
print("PASS All model_loader imports OK")

import os, neurocuda
hub_dir = os.path.expanduser("~/.cache/neurocuda/hub")
if os.path.isdir(hub_dir):
    models = os.listdir(hub_dir)
    print(f"INFO Cached models: {models}")
else:
    print("INFO No cached models - will download on first use")
EOF

echo ""
echo "=========================================="
echo "  STEP 5: Node script syntax check"
echo "=========================================="
python3 -c "
import ast
for f in ['snn_inference_node.py', 'snn_control_node.py', 'spike_viz.py', 'model_loader.py']:
    path = f'src/neurocuda_ros2/neurocuda_ros2/{f}'
    ast.parse(open(path).read())
    print(f'PASS Syntax OK: {f}')
"

echo ""
echo "=========================================="
echo "  ALL TESTS PASSED ✅"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✅ Build: neurocuda_msgs + neurocuda_ros2"
echo "  ✅ Messages: SnnDetection, SnnSpikeEvent, SnnStatus"
echo "  ✅ Python: import, create, populate, validate"
echo "  ✅ ros2 interface: all 3 show correctly"
echo "  ✅ Syntax: all 4 node files valid Python"
echo "  🔲 Runtime: needs ROS2 graph + camera (manual test)"
