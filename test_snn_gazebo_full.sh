#!/bin/bash
# Full pipeline: Gazebo OGRE2 camera → SNN detection — state-verified lifecycle
#
# Pipeline: Gazebo (ogre2, 28x28 mono8) → ros_gz_bridge → /camera/image_raw
#           → topic_tools relay → /camera/image → SNN → /snn/detections
#
# KEY FIX: Polls lifecycle get_state before each transition.
#           No fixed sleep-based timing. Retries with backoff.
set -e
source /opt/ros/jazzy/setup.bash
cd /neurocuda_ws
colcon build --packages-select neurocuda_msgs neurocuda_ros2 2>&1 | tail -3
source install/setup.bash
export QT_QPA_PLATFORM=offscreen
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0

pass() { echo -e "${GREEN}✅ PASS${NC}: $1"; PASS=$((PASS+1)); }
fail() { echo -e "${RED}❌ FAIL${NC}: $1"; FAIL=$((FAIL+1)); }
warn() { echo -e "${YELLOW}⚠️  WARN${NC}: $1"; }
info() { echo -e "${CYAN}📌${NC} $1"; }

# ============================================================================
# State-polling helper — waits until lifecycle node reaches expected state
# ============================================================================
wait_for_state() {
    local node=$1
    local expected_id=$2
    local timeout=${3:-60}
    local label=${4:-"STATE_$expected_id"}
    local start=$(date +%s)

    info "Waiting for /$node → $label (timeout=${timeout}s)..."

    while true; do
        local elapsed=$(($(date +%s) - start))
        if [ $elapsed -ge $timeout ]; then
            fail "$node did not reach $label within ${timeout}s"
            return 1
        fi

        local result
        result=$(ros2 service call "/$node/get_state" lifecycle_msgs/srv/GetState "{}" 2>/dev/null || echo "")

        if [ -z "$result" ]; then
            sleep 0.5
            continue
        fi

        local current_id
        current_id=$(echo "$result" | grep "id=" | head -1 | sed 's/.*id=\([0-9]*\).*/\1/')

        if [ -z "$current_id" ]; then
            sleep 0.5
            continue
        fi

        if [ "$current_id" -eq "$expected_id" ]; then
            echo -e "${GREEN}✅ $node reached $label${NC} (after ${elapsed}s)"
            return 0
        fi

        # Check for error states
        if [ "$current_id" -eq 4 ]; then  # FINALIZED
            fail "$node entered FINALIZED (error state)"
            return 1
        fi

        # Progress indicator (every 5s)
        if [ $((elapsed % 5)) -eq 0 ] && [ $elapsed -gt 0 ]; then
            info "  Still waiting... state_id=$current_id (${elapsed}s elapsed)"
        fi

        sleep 0.5
    done
}

# ============================================================================
# Call lifecycle transition with retry
# ============================================================================
call_transition() {
    local node=$1
    local transition_id=$2
    local transition_label=$3
    local max_retries=${4:-3}

    for attempt in $(seq 1 $max_retries); do
        info "Calling /$node/change_state → $transition_label (attempt $attempt/$max_retries)"

        local result
        result=$(ros2 service call "/$node/change_state" lifecycle_msgs/srv/ChangeState \
            "{transition: {id: $transition_id, label: '$transition_label'}}" 2>&1) || true

        if echo "$result" | grep -q "success=True"; then
            echo -e "${GREEN}✅ $transition_label returned success=True${NC}"
            return 0
        elif echo "$result" | grep -q "success=False"; then
            warn "$transition_label returned success=False — will retry after delay"
            sleep 2
        else
            warn "$transition_label: unexpected response: $result"
            sleep 2
        fi
    done

    fail "$transition_label FAILED after $max_retries attempts"
    return 1
}

# ============================================================================
# STEP 1: Start Gazebo with OGRE2 camera world (headless)
# ============================================================================
echo ""
echo "=========================================="
echo "  STEP 1: Gazebo OGRE2 headless camera"
echo "=========================================="

WORLD_FILE="/neurocuda_ws/src/neurocuda_ros2/worlds/ogre2_camera_28.sdf"
if [ ! -f "$WORLD_FILE" ]; then
    fail "OGRE2 world file not found: $WORLD_FILE"
    exit 1
fi

info "Starting Gazebo with OGRE2 render engine (28x28 mono8 camera)..."
gz sim -s -r --headless-rendering "$WORLD_FILE" &
GZ_PID=$!
sleep 3

# Verify Gazebo is running
if kill -0 $GZ_PID 2>/dev/null; then
    pass "Gazebo server running (PID $GZ_PID)"
else
    fail "Gazebo server failed to start"
    exit 1
fi

# ============================================================================
# STEP 2: Bridge camera from Gazebo to ROS2
# ============================================================================
echo ""
echo "  STEP 2: ROS-GZ bridge (camera → ROS2)"
echo ""

# Bridge the ogre2 camera sensor to ROS2
# Gazebo topic: /world/test/model/cam/link/link/sensor/c/image
# ROS2 topic: /camera/image_raw
ros2 run ros_gz_bridge parameter_bridge \
    "/world/test/model/cam/link/link/sensor/c/image@sensor_msgs/msg/Image@gz.msgs.Image" &
BRIDGE_PID=$!
sleep 3

# Verify camera topic appears
if ros2 topic list 2>/dev/null | grep -q "/world/test/model/cam/link/link/sensor/c/image"; then
    pass "Camera topic bridged from Gazebo"
    ros2 topic info "/world/test/model/cam/link/link/sensor/c/image" 2>/dev/null || true
else
    warn "Camera topic not yet visible — checking gz topics..."
    gz topic -l 2>/dev/null | grep -i cam || true
fi

# ============================================================================
# STEP 3: Relay camera to /camera/image (what SNN subscribes to)
# ============================================================================
echo ""
echo "  STEP 3: Relay → /camera/image"
echo ""

# Use topic_tools relay to remap the long gz topic to clean /camera/image
ros2 run topic_tools relay \
    "/world/test/model/cam/link/link/sensor/c/image" \
    "/camera/image" &
RELAY_PID=$!
sleep 2

if ros2 topic list 2>/dev/null | grep -q "/camera/image"; then
    pass "Relay active: /camera/image published"
    CAM_INFO=$(timeout 5 ros2 topic info /camera/image 2>&1 || echo "")
    echo "    $CAM_INFO"
else
    warn "/camera/image not yet visible — relay may need more time"
fi

# ============================================================================
# STEP 4: Start SNN inference node + lifecycle manager
# ============================================================================
echo ""
echo "  STEP 4: Start SNN inference node (lifecycle)"
echo ""

python3 -m neurocuda_ros2.snn_inference_node --ros-args \
    -p model:="neurocuda/mlp-mnist-snn" \
    -p device:="cpu" \
    -p input_type:="image" \
    -p T:=8 \
    -p camera_topic:="/camera/image" &
SNN_PID=$!
sleep 2

if kill -0 $SNN_PID 2>/dev/null; then
    pass "SNN inference node running (PID $SNN_PID)"
else
    fail "SNN inference node failed to start"
    exit 1
fi

# Wait for lifecycle services to appear
info "Waiting for lifecycle services..."
for i in $(seq 1 20); do
    if ros2 service list 2>/dev/null | grep -q "/snn_inference/change_state"; then
        pass "Lifecycle services available"
        break
    fi
    sleep 1
done

# ============================================================================
# STEP 5: Configure — with state verification
# ============================================================================
echo ""
echo "  STEP 5: Configure SNN (Unconfigured → Inactive)"
echo ""

# Check initial state
info "Checking initial state..."
INIT_STATE=$(ros2 service call /snn_inference/get_state lifecycle_msgs/srv/GetState "{}" 2>/dev/null | grep "id=" | head -1 | sed 's/.*id=\([0-9]*\).*/\1/' || echo "unknown")
info "Initial state id=$INIT_STATE"

# Lifecycle state IDs: 0=UNKNOWN, 1=UNCONFIGURED, 2=INACTIVE, 3=ACTIVE, 4=FINALIZED
if [ "$INIT_STATE" = "2" ]; then
    info "Already INACTIVE — skipping configure"
elif [ "$INIT_STATE" = "3" ]; then
    info "Already ACTIVE — skipping configure"
elif call_transition "snn_inference" 1 "configure"; then
    # Verify state reached INACTIVE (id=2)
    wait_for_state "snn_inference" 2 120 "INACTIVE"
else
    fail "Configure failed"
fi

# ============================================================================
# STEP 6: Activate — with state verification
# ============================================================================
echo ""
echo "  STEP 6: Activate SNN (Inactive → Active)"
echo ""

CURRENT_STATE=$(ros2 service call /snn_inference/get_state lifecycle_msgs/srv/GetState "{}" 2>/dev/null | grep "id=" | head -1 | sed 's/.*id=\([0-9]*\).*/\1/' || echo "unknown")
info "Current state before activate: id=$CURRENT_STATE"

if [ "$CURRENT_STATE" = "3" ]; then
    info "Already ACTIVE — skipping activate"
elif [ "$CURRENT_STATE" = "2" ]; then
    if call_transition "snn_inference" 3 "activate"; then
        wait_for_state "snn_inference" 3 60 "ACTIVE"
    else
        fail "Activate failed"
    fi
else
    fail "Cannot activate from state $CURRENT_STATE (expected INACTIVE=1)"
fi

# ============================================================================
# STEP 7: Check ROS2 graph
# ============================================================================
echo ""
echo "  STEP 7: ROS2 Graph"
echo ""

echo "Nodes:"
ros2 node list 2>/dev/null || echo "  (none)"
echo ""
echo "SNN Topics:"
ros2 topic list 2>/dev/null | grep -E "/snn/|/camera/" || echo "  (no SNN topics yet)"

# ============================================================================
# STEP 8: Verify SNN topics exist with correct types
# ============================================================================
echo ""
echo "  STEP 8: Verify SNN Topics"
echo ""

for topic in /snn/detections /snn/spikes /snn/sparsity /snn/status; do
    if ros2 topic list 2>/dev/null | grep -q "$topic"; then
        TYPE=$(timeout 3 ros2 topic info "$topic" 2>/dev/null | grep "Type:" | awk '{print $2}' || echo "unknown")
        echo -e "    ${GREEN}✅${NC} $topic ($TYPE)"
    else
        echo -e "    ${YELLOW}⚠️${NC}  $topic — NOT YET PUBLISHED (SNN may need camera data first)"
    fi
done

# ============================================================================
# STEP 9: Capture actual detection message (multiple attempts)
# ============================================================================
echo ""
echo "  STEP 9: Capture /snn/detections"
echo ""

DET_CAPTURED=false
for attempt in $(seq 1 5); do
    info "Detection capture attempt $attempt/5..."

    # Check if we have publishers on /camera/image
    PUB_COUNT=$(timeout 3 ros2 topic info /camera/image 2>/dev/null | grep "Publisher count:" | awk '{print $3}' || echo "0")
    info "  /camera/image publisher count: $PUB_COUNT"

    # Check camera frame format
    CAM_FMT=$(timeout 5 ros2 topic echo /camera/image --once 2>&1 | head -8 || echo "")
    if echo "$CAM_FMT" | grep -q "height"; then
        info "  Camera frame flowing:"
        echo "$CAM_FMT" | head -5
    else
        warn "  No camera frame received yet"
    fi

    # Try to capture a detection
    DET=$(timeout 8 ros2 topic echo /snn/detections --once 2>&1 || echo "TIMEOUT")

    if echo "$DET" | grep -qE "class_id|class_name"; then
        echo ""
        echo -e "${GREEN}=========================================="
        echo "  ✅ DETECTION CAPTURED"
        echo -e "==========================================${NC}"
        echo "$DET"
        echo ""
        DET_CAPTURED=true
        break
    elif echo "$DET" | grep -q "TIMEOUT"; then
        warn "  Timeout — no detection published yet (SNN may not be receiving frames)"
    else
        warn "  Unexpected response: $(echo "$DET" | head -3)"
    fi

    sleep 2
done

if [ "$DET_CAPTURED" = false ]; then
    echo ""
    echo -e "${RED}=========================================="
    echo "  ❌ NO DETECTION CAPTURED after 5 attempts"
    echo -e "==========================================${NC}"
    echo ""
    echo "Debug:"
    echo "  - Camera topic:"
    ros2 topic info /camera/image 2>/dev/null || echo "    (no info)"
    echo "  - SNN node state:"
    ros2 service call /snn_inference/get_state lifecycle_msgs/srv/GetState "{}" 2>/dev/null || echo "    (no response)"
    echo "  - Recent SNN logs (last 10 lines):"
    FAIL=1
fi

# ============================================================================
# STEP 10: Honest result summary
# ============================================================================
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY"
echo "=========================================="
echo ""
echo "Pipeline tested:"
echo "  Gazebo (OGRE2, 28×28 mono8) → ros_gz_bridge → relay → /camera/image"
echo "  → SNN (mlp-mnist-snn, 97.4% acc) → /snn/detections"
echo ""
echo "⚠️  SEMANTIC NOTE: The MLP-MNIST model outputs digit classes (0-9)."
echo "   The Gazebo scene contains a red box — not handwritten digits."
echo "   Classification labels WILL NOT be semantically meaningful."
echo "   This is an INFRASTRUCTURE verification test, not an accuracy test."
echo ""

if [ "$DET_CAPTURED" = true ]; then
    echo -e "${GREEN}✅ PIPELINE VERIFIED: Camera → SNN → Detection (end-to-end)${NC}"
    echo "   Label: infrastructure-only (semantics are MNIST digits on 3D scene)"
else
    echo -e "${RED}❌ PIPELINE BLOCKED: Detection not captured${NC}"
    echo "   See debug info above for root cause"
fi

echo ""
echo "Verdicts: PASS=$PASS FAIL=$FAIL"

# ============================================================================
# Cleanup
# ============================================================================
echo ""
echo "  Cleaning up..."
kill $SNN_PID $RELAY_PID $BRIDGE_PID $GZ_PID 2>/dev/null || true
sleep 2
echo "  Done."

exit $FAIL
