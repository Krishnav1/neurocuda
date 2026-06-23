#!/bin/bash
# NeuroCUDA ROS2 — Quick Verification Test
# Run this after setup to verify everything works.
# Usage: bash scripts/quick_test.sh

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
PASS=0; FAIL=0
pass() { echo -e "${GREEN}✅${NC} $1"; PASS=$((PASS+1)); }
fail() { echo -e "${RED}❌${NC} $1"; FAIL=$((FAIL+1)); }
info() { echo -e "${CYAN}▶${NC} $1"; }

echo "============================================"
echo "  NeuroCUDA ROS2 — Quick Test"
echo "============================================"
echo ""

# 1. Environment
info "Checking environment..."
source /opt/ros/jazzy/setup.bash 2>/dev/null && pass "ROS2 Jazzy sourced" || { fail "ROS2 Jazzy not found"; exit 1; }
source /neurocuda_ws/install/setup.bash 2>/dev/null && pass "Workspace built" || fail "Workspace not built (run: colcon build)"
python3 -c "import neurocuda" 2>/dev/null && pass "NeuroCUDA installed" || fail "NeuroCUDA not installed"
python3 -c "import rclpy" 2>/dev/null && pass "rclpy available" || fail "rclpy not available"
echo ""

# 2. Messages
info "Checking custom messages..."
ros2 interface show neurocuda_msgs/msg/SnnDetection 2>/dev/null | grep -q class_id && pass "SnnDetection message" || fail "SnnDetection message"
ros2 interface show neurocuda_msgs/msg/SnnSpikeEvent 2>/dev/null | grep -q layer_name && pass "SnnSpikeEvent message" || fail "SnnSpikeEvent message"
ros2 interface show neurocuda_msgs/msg/SnnStatus 2>/dev/null | grep -q model_name && pass "SnnStatus message" || fail "SnnStatus message"
echo ""

# 3. Launch files
info "Checking launch files..."
for launch in infer dashboard benchmark benchmark_replay record replay demo_synthetic demo_gazebo_headless; do
    ros2 launch neurocuda_ros2 ${launch}.launch.py --show-arguments 2>/dev/null >/dev/null && pass "${launch}.launch.py" || fail "${launch}.launch.py"
done
echo ""

# 4. SNN node lifecycle
info "Testing SNN node lifecycle..."
python3 -m neurocuda_ros2.snn_inference_node --ros-args \
    -p model:=neurocuda/mlp-mnist-snn -p device:=cpu \
    -p camera_topic:=/camera/image &
SNN_PID=$!
sleep 4

ros2 service list 2>/dev/null | grep -q /snn_inference/change_state && pass "Lifecycle services available" || { fail "Lifecycle services"; kill $SNN_PID 2>/dev/null; exit 1; }

ros2 service call /snn_inference/change_state lifecycle_msgs/srv/ChangeState \
    "{transition: {id: 1, label: configure}}" 2>&1 | grep -q True && pass "Configured" || fail "Configure"

sleep 2
ros2 service call /snn_inference/change_state lifecycle_msgs/srv/ChangeState \
    "{transition: {id: 3, label: activate}}" 2>&1 | grep -q True && pass "Activated" || fail "Activate"

sleep 2
ros2 topic list 2>/dev/null | grep -q /snn/detections && pass "/snn/detections publishing" || fail "/snn/detections"
ros2 topic list 2>/dev/null | grep -q /snn/sparsity && pass "/snn/sparsity publishing" || fail "/snn/sparsity"
ros2 topic list 2>/dev/null | grep -q /snn/spikes && pass "/snn/spikes publishing" || fail "/snn/spikes"
ros2 topic list 2>/dev/null | grep -q /snn/status && pass "/snn/status publishing" || fail "/snn/status"

kill $SNN_PID 2>/dev/null
echo ""

# 5. Summary
echo "============================================"
echo "  Results: ${GREEN}$PASS passed${NC} / ${RED}$FAIL failed${NC}"
echo "============================================"
if [ $FAIL -eq 0 ]; then
    echo ""
    echo "  ✅ All tests passed! Pipeline is ready."
    echo ""
    echo "  Next steps:"
    echo "    ros2 launch neurocuda_ros2 dashboard.launch.py"
    echo "    ros2 launch neurocuda_ros2 benchmark.launch.py"
    echo "    ros2 launch neurocuda_ros2 demo_gazebo_headless.launch.py"
else
    echo "  ❌ Some tests failed. See above."
    exit 1
fi
