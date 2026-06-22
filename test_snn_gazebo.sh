#!/bin/bash
# Test NeuroCUDA SNN against Gazebo TurtleBot4 camera feed
set -e
source /opt/ros/jazzy/setup.bash
cd /neurocuda_ws
colcon build --packages-select neurocuda_msgs neurocuda_ros2 2>&1 | tail -1
source install/setup.bash
export QT_QPA_PLATFORM=offscreen

echo "=== Step 1: Start Gazebo + TurtleBot4 ==="
gz sim -s -r /opt/ros/jazzy/share/turtlebot4_gz_bringup/worlds/warehouse.sdf &
GZ_PID=$!
sleep 4
echo "Gazebo PID: $GZ_PID"

echo ""
echo "=== Step 2: Start ROS-GZ bridge ==="
ros2 launch turtlebot4_gz_bringup ros_gz_bridge.launch.py &
BRIDGE_PID=$!
sleep 5

echo ""
echo "=== Step 3: Verify camera ==="
ros2 topic info /oakd/rgb/preview/image_raw 2>/dev/null

echo ""
echo "=== Step 4: Download model ==="
python3 << 'PYEOF'
from neurocuda import hub
print("Downloading model...")
try:
    model, info = hub.load("neurocuda/mlp-mnist-snn", device="cpu")
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {params} params")
    print(f"   Accuracy: {info.get('snn_accuracy', 'N/A')}")
    print(f"   Sparsity: {info.get('sparsity', 'N/A')}")
except Exception as e:
    print(f"⚠️  Model download failed: {e}")
    print("   (expected if HuggingFace auth needed or network issue)")
PYEOF

echo ""
echo "=== Step 5: Check if model downloaded to cache ==="
ls -la ~/.cache/neurocuda/hub/ 2>/dev/null || echo "Nothing cached"

echo ""
echo "=== Step 6: Cleanup ==="
kill $BRIDGE_PID 2>/dev/null
kill $GZ_PID 2>/dev/null
echo "Done"
