#!/usr/bin/env python3
"""
NeuroCUDA SNN Control Node for ROS2.

Subscribes to:
  /robot/state (any Float32MultiArray or Odometry) — robot state

Publishes:
  /cmd_vel (geometry_msgs/Twist) — velocity commands
  /snn/action (std_msgs/String) — action label
  /snn/q_values — Q-values for all actions

Parameters:
  model: Model name (default: "neurocuda/dqn-cartpole-snn")
  T: Timesteps for SNN temporal integration (default: 16)
  device: "cuda" or "cpu" (auto-detected)
  action_mode: "discrete" or "continuous"
  num_actions: Number of discrete actions (for DQN)
  publish_period: Seconds between control updates (default: 0.05)

Usage:
  ros2 run neurocuda_ros2 snn_control --ros-args -p model:=dqn-cartpole-snn
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.timer import Timer

import numpy as np
import torch

from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32MultiArray

# State message types we support
try:
    from nav_msgs.msg import Odometry
    HAS_ODOM = True
except ImportError:
    HAS_ODOM = False


class SNNControlNode(Node):
    """ROS2 node that uses a spiking DQN for robot control."""

    def __init__(self):
        super().__init__("snn_control")

        # --- Parameters ---
        self.declare_parameter("model", "neurocuda/dqn-cartpole-snn")
        self.declare_parameter("T", 16)
        self.declare_parameter("device", "auto")
        self.declare_parameter("action_mode", "discrete")
        self.declare_parameter("num_actions", 2)
        self.declare_parameter("publish_period", 0.05)  # 20 Hz

        model_name = self.get_parameter("model").value
        self.T = self.get_parameter("T").value
        device_opt = self.get_parameter("device").value
        self.action_mode = self.get_parameter("action_mode").value
        self.num_actions = self.get_parameter("num_actions").value
        period = self.get_parameter("publish_period").value

        if device_opt == "auto":
            device_opt = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Load SNN Model ---
        from neurocuda_ros2.model_loader import ModelLoader

        self.get_logger().info(f"Loading SNN control model: {model_name}")
        try:
            self.model_loader = ModelLoader(
                model_name, device=device_opt
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

        self.get_logger().info(
            f"Control model loaded: {self.model_loader.num_params:,} params | "
            f"Actions: {self.num_actions} | Mode: {self.action_mode}"
        )

        # --- Publishers ---
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.action_pub = self.create_publisher(String, "/snn/action", 10)
        self.qvalues_pub = self.create_publisher(Float32MultiArray, "/snn/q_values", 10)

        # --- Subscribers ---
        # Generic state array (works with any robot)
        self.state_sub = self.create_subscription(
            Float32MultiArray, "/robot/state", self.state_callback, 10
        )

        if HAS_ODOM:
            self.odom_sub = self.create_subscription(
                Odometry, "/odom", self.odom_callback, 10
            )

        # --- Timer for periodic control ---
        self.current_state = None
        self.control_timer = self.create_timer(period, self.control_step)

        # Action mapping
        self.action_names = self._get_action_names(model_name)

        self.get_logger().info("SNN Control Node ready")

    def _get_action_names(self, model_name):
        """Get action names based on model."""
        if "cartpole" in model_name:
            return ["left", "right"]
        return [f"action_{i}" for i in range(self.num_actions)]

    # ------------------------------------------------------------------
    # State Callbacks
    # ------------------------------------------------------------------
    def state_callback(self, msg):
        """Store latest robot state from Float32MultiArray."""
        self.current_state = np.array(msg.data, dtype=np.float32)

    def odom_callback(self, msg):
        """Store state from Odometry message."""
        # Extract: x, vx, theta, theta_dot (CartPole-like state)
        self.current_state = np.array([
            msg.pose.pose.position.x,
            msg.twist.twist.linear.x,
            msg.pose.pose.orientation.z,  # approximate theta
            msg.twist.twist.angular.z,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Control Step (periodic)
    # ------------------------------------------------------------------
    def control_step(self):
        """Run SNN inference and publish action."""
        if self.current_state is None:
            return

        try:
            # Reset SNN state for each control step
            self.model_loader.reset_state()

            # Convert state → tensor
            state_tensor = torch.from_numpy(self.current_state).float().unsqueeze(0)
            state_tensor = state_tensor.to(self.model_loader.device)

            # Run SNN DQN — accumulates Q-values over T timesteps
            with torch.no_grad():
                q_values = self.model_loader.model(state_tensor)

            # Select action (greedy — no exploration in deployment)
            q_vals_np = q_values[0].cpu().numpy()
            action = int(np.argmax(q_vals_np))

            # --- Publish Action ---
            action_msg = String()
            action_name = self.action_names[action] if action < len(self.action_names) else str(action)
            action_msg.data = f"action={action_name} q_values={q_vals_np.tolist()}"
            self.action_pub.publish(action_msg)

            # --- Publish Q-values ---
            q_msg = Float32MultiArray()
            q_msg.data = q_vals_np.tolist()
            self.qvalues_pub.publish(q_msg)

            # --- Publish Velocity Command ---
            cmd = Twist()
            if self.action_mode == "discrete":
                if self.num_actions == 2:  # CartPole-style: left/right
                    force = 10.0 if action == 0 else -10.0
                    cmd.linear.x = force
                elif self.num_actions == 4:  # Up/down/left/right
                    directions = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
                    cmd.linear.x, cmd.linear.y = directions[action]
                else:
                    cmd.linear.x = float(action)
            else:
                # Continuous: Q-values are direct control signals
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
        node.get_logger().info("Shutting down SNN Control Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
