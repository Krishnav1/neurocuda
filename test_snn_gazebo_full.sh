#!/bin/bash
# Complete pipeline: Gazebo → TurtleBot4 camera → NeuroCUDA SNN → detections
set -e
source /opt/ros/jazzy/setup.bash
cd /neurocuda_ws
colcon build --packages-select neurocuda_msgs neurocuda_ros2 2>&1 | tail -1
source install/setup.bash
export QT_QPA_PLATFORM=offscreen

echo "=========================================="
echo "  STEP 1: Start Gazebo + TurtleBot4"
echo "=========================================="
gz sim -s -r /opt/ros/jazzy/share/turtlebot4_gz_bringup/worlds/warehouse.sdf &
GZ_PID=$!
sleep 4
echo "Gazebo: PID $GZ_PID"

echo ""
echo "  STEP 2: Start ROS-GZ bridge"
echo ""
ros2 launch turtlebot4_gz_bringup ros_gz_bridge.launch.py &
BRIDGE_PID=$!
sleep 6

echo ""
echo "  STEP 3: Verify camera publishing"
echo ""
ros2 topic info /oakd/rgb/preview/image_raw

echo ""
echo "  STEP 4: Start camera_sim ALSO (for /camera/image topic)"
echo "          This feeds synthetic images while Gazebo feeds its own camera"
echo ""
# Start camera_sim to feed /camera/image which is what SNN expects
python3 -m neurocuda_ros2.camera_sim --ros-args -p pattern:=airplane -p rate:=2.0 -p resolution:='[64,64]' &
CAM_PID=$!
sleep 2
echo "Camera sim: PID $CAM_PID"

echo ""
echo "  STEP 5: Start SNN inference node"
echo "          SNN subscribes to /camera/image (from camera_sim)"
echo ""
python3 -m neurocuda_ros2.snn_inference_node --ros-args \
  -p model:="neurocuda/mlp-mnist-snn" \
  -p device:="cpu" \
  -p input_type:="image" \
  -p T:=8 &
SNN_PID=$!
sleep 3

echo ""
echo "  STEP 6: Configure & Activate SNN (lifecycle)"
echo ""
ros2 service call /snn_inference/change_state lifecycle_msgs/srv/ChangeState \
  "{transition: {id: 1, label: 'configure'}}" 2>/dev/null || echo "(may already be configured)"
sleep 2
ros2 service call /snn_inference/change_state lifecycle_msgs/srv/ChangeState \
  "{transition: {id: 3, label: 'activate'}}" 2>/dev/null || echo "(may already be active)"
sleep 2

echo ""
echo "  STEP 7: Check ROS2 graph"
echo ""
echo "Nodes:"
ros2 node list
echo ""
echo "Topics:"
ros2 topic list | head -20

echo ""
echo "  STEP 8: Verify SNN topics"
echo ""
for topic in /snn/detections /snn/spikes /snn/sparsity /snn/status; do
    if ros2 topic list | grep -q "$topic"; then
        TYPE=$(ros2 topic info "$topic" 2>/dev/null | grep "Type:" | awk '{print $2}')
        echo "    ✅ $topic ($TYPE)"
    else
        echo "    ❌ $topic NOT FOUND"
    fi
done

echo ""
echo "  STEP 9: Capture a detection"
echo ""
DET=$(timeout 10 ros2 topic echo /snn/detections --once 2>&1 || echo "no_detection_yet")
if echo "$DET" | grep -q "class"; then
    echo "✅ DETECTION PUBLISHED:"
    echo "$DET" | head -15
elif echo "$DET" | grep -q "no_detection"; then
    echo "⚠️  No detection yet (SNN may not have processed image)"
else
    echo "SNN output: $DET" | head -5
fi

echo ""
echo "  STEP 10: Cleanup"
echo ""
kill $SNN_PID $CAM_PID $BRIDGE_PID $GZ_PID 2>/dev/null
sleep 2
echo "Done"
