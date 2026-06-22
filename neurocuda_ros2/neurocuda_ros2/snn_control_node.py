#!/usr/bin/env python3
"""
NeuroCUDA SNN Control Node — ROS2 Managed Lifecycle Node.

Uses a spiking DQN for robot control. Follows the lifecycle pattern:
  Unconfigured → Configure (load DQN) → Activate (start control loop)
  → Deactivate (pause) → Cleanup (free GPU)

Subscribes to:
  /robot/state (std_msgs/Float32MultiArray) — robot state
  /odom (nav_msgs/Odometry) — odometry state

Publishes:
  /cmd_vel (geometry_msgs/Twist) — velocity commands
  /snn/action (neurocuda_msgs/SnnStatus) — action + model info
  /snn/q_values (std_msgs/Float32MultiArray) — Q-values for all actions

Usage:
  ros2 launch neurocuda_ros2 control.launch.py model:=dqn-cartpole-snn
"""

import rclpy
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn

import numpy as np
import torch

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from neurocuda_msgs.msg import SnnStatus

try:
    from nav_msgs.msg import Odometry
    HAS_ODOM = True
except ImportError:
    HAS_ODOM = False


class SNNControlNode(Node):
    """Managed lifecycle node for spiking DQN robot control."""

    def __init__(self, node_name="snn_control"):
        super().__init__(node_name=node_name)
        self.model_loader = None
        self.control_timer = None
        self.current_state = None
        self.action_names = []
        self._num_actions = 2
        self._action_mode = "discrete"

    # ==================================================================
    # LIFECYCLE: on_configure
    # ==================================================================
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("⚙️  Configuring SNN control node...")

        try:
            self.declare_parameter("model", "neurocuda/dqn-cartpole-snn")
            self.declare_parameter("T", 16)
            self.declare_parameter("device", "auto")
            self.declare_parameter("action_mode", "discrete")
            self.declare_parameter("num_actions", 2)
            self.declare_parameter("publish_period", 0.05)

            model_name = self.get_parameter("model").value
            T_val = self.get_parameter("T").value
            device_opt = self.get_parameter("device").value
            self._action_mode = self.get_parameter("action_mode").value
            self._num_actions = self.get_parameter("num_actions").value
            period = self.get_parameter("publish_period").value

            if device_opt == "auto":
                device_opt = "cuda" if torch.cuda.is_available() else "cpu"

            # Load SNN DQN model
            from neurocuda_ros2.model_loader import ModelLoader
            self.get_logger().info(f"  Loading DQN model: {model_name}")
            self.model_loader = ModelLoader(model_name, device=device_opt)

            self.get_logger().info(
                f"  ✅ Control model loaded: {self.model_loader.num_params:,} params | "
                f"Actions: {self._num_actions} | Mode: {self._action_mode}"
            )

            # Lifecycle publishers
            self.cmd_pub = self.create_lifecycle_publisher(Twist, "/cmd_vel", 10)
            self.action_pub = self.create_lifecycle_publisher(SnnStatus, "/snn/action", 10)
            self.qvalues_pub = self.create_lifecycle_publisher(Float32MultiArray, "/snn/q_values", 10)

            # Subscriptions
            self.state_sub = self.create_subscription(
                Float32MultiArray, "/robot/state", self.state_callback, 10)
            if HAS_ODOM:
                self.odom_sub = self.create_subscription(
                    Odometry, "/odom", self.odom_callback, 10)

            # Action names
            self.action_names = self._get_action_names(model_name)
            self.T = T_val
            self._period = period

            self.get_logger().info("  ✅ Configured — ready to activate")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"  ❌ Configure failed: {e}")
            return TransitionCallbackReturn.FAILURE

    # ==================================================================
    # LIFECYCLE: on_activate
    # ==================================================================
    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("▶️  Activating — SNN control loop starting")
        self.control_timer = self.create_timer(self._period, self.control_step)
        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # LIFECYCLE: on_deactivate
    # ==================================================================
    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("⏸️  Deactivating — pausing control loop")
        if self.control_timer:
            self.destroy_timer(self.control_timer)
            self.control_timer = None
        if self.model_loader:
            self.model_loader.reset_state()
        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # LIFECYCLE: on_cleanup
    # ==================================================================
    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("🧹 Cleaning up...")
        try:
            self.destroy_publisher(self.cmd_pub)
            self.destroy_publisher(self.action_pub)
            self.destroy_publisher(self.qvalues_pub)
        except Exception:
            pass
        try:
            self.destroy_subscription(self.state_sub)
            if HAS_ODOM:
                self.destroy_subscription(self.odom_sub)
        except Exception:
            pass
        if self.model_loader:
            del self.model_loader
            self.model_loader = None
        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # LIFECYCLE: on_shutdown
    # ==================================================================
    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("🛑 Shutting down SNN Control Node")
        self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # LIFECYCLE: on_error
    # ==================================================================
    def on_error(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().error("⚠️  Error — attempting recovery")
        self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_action_names(self, model_name):
        if model_name and "cartpole" in model_name:
            return ["left", "right"]
        return [f"action_{i}" for i in range(self._num_actions)]

    # ------------------------------------------------------------------
    # State Callbacks
    # ------------------------------------------------------------------
    def state_callback(self, msg):
        self.current_state = np.array(msg.data, dtype=np.float32)

    def odom_callback(self, msg):
        self.current_state = np.array([
            msg.pose.pose.position.x,
            msg.twist.twist.linear.x,
            msg.pose.pose.orientation.z,
            msg.twist.twist.angular.z,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Control Step
    # ------------------------------------------------------------------
    def control_step(self):
        if self.current_state is None or self.model_loader is None:
            return

        try:
            self.model_loader.reset_state()
            state_tensor = torch.from_numpy(self.current_state).float().unsqueeze(0)
            state_tensor = state_tensor.to(self.model_loader.device)

            with torch.no_grad():
                q_values = self.model_loader.model(state_tensor)

            spike_stats = self.model_loader._get_spike_stats()
            q_vals_np = q_values[0].cpu().numpy()
            action = int(np.argmax(q_vals_np))

            # Publish action
            action_name = self.action_names[action] if action < len(self.action_names) else str(action)
            action_msg = SnnStatus()
            action_msg.model_name = self.model_loader.model_name
            action_msg.task = f"action={action_name}"
            action_msg.architecture = self.model_loader._describe_architecture()
            action_msg.accuracy = float(self.model_loader.accuracy) if isinstance(self.model_loader.accuracy, (int, float)) else 0.0
            action_msg.total_params = self.model_loader.num_params
            action_msg.neuron_count = self.model_loader.if_count + self.model_loader.lif_count
            action_msg.device = str(self.model_loader.device)
            action_msg.avg_sparsity = float(spike_stats.get("sparsity", 0.0))
            action_msg.inference_time_ms = 0.0
            self.action_pub.publish(action_msg)

            # Publish Q-values
            q_msg = Float32MultiArray()
            q_msg.data = q_vals_np.tolist()
            self.qvalues_pub.publish(q_msg)

            # Publish velocity command
            cmd = Twist()
            if self._action_mode == "discrete":
                if self._num_actions == 2:
                    force = 10.0 if action == 0 else -10.0
                    cmd.linear.x = force
                elif self._num_actions == 4:
                    directions = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
                    cmd.linear.x, cmd.linear.y = directions[action]
                else:
                    cmd.linear.x = float(action)
            else:
                cmd.linear.x = float(q_vals_np[0])
                cmd.angular.z = float(q_vals_np[1]) if len(q_vals_np) > 1 else 0.0

            self.cmd_pub.publish(cmd)

        except Exception as e:
            self.get_logger().error(f"Control error: {e}", throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = SNNControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
