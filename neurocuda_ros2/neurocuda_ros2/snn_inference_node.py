#!/usr/bin/env python3
"""
NeuroCUDA SNN Inference Node for ROS2.

Subscribes to:
  /camera/image (sensor_msgs/Image) — regular camera
  /dvs/events (event_camera_msgs/EventArray) — event camera

Publishes:
  /detections (custom) — classification results
  /spikes (custom) — spike activity statistics
  /snn/debug_image — visualization overlay

Parameters:
  model: Model name (default: "neurocuda/cnn-nmnist-snn")
  input_type: "image" or "events" (auto-detected)
  T: Timesteps for event accumulation (default: 16)
  device: "cuda" or "cpu" (auto-detected)
  hardware_target: "gpu", "cpu", "loihi", "fpga"

Usage:
  ros2 run neurocuda_ros2 snn_infer --ros-args -p model:=cnn-nmnist-snn
  ros2 run neurocuda_ros2 snn_infer --ros-args -p input_type:=events
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

# Message types
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from neurocuda_msgs.msg import SnnDetection, SnnSpikeEvent, SnnStatus
import numpy as np

# NeuroCUDA
from neurocuda_ros2.model_loader import (
    ModelLoader, image_to_tensor, events_to_tensor,
    detection_to_msg, spike_stats_to_msg, status_to_msg
)

# Event camera messages (optional — works without them)
try:
    from event_camera_msgs.msg import EventArray
    HAS_EVENT_MSGS = True
except ImportError:
    HAS_EVENT_MSGS = False


class SNNInferenceNode(Node):
    """ROS2 node that runs spiking neural network inference."""

    def __init__(self):
        super().__init__("snn_inference")

        # --- Parameters ---
        self.declare_parameter("model", "neurocuda/cnn-nmnist-snn")
        self.declare_parameter("input_type", "auto")  # auto/image/events
        self.declare_parameter("T", 16)  # timesteps for event accumulation
        self.declare_parameter("device", "auto")
        self.declare_parameter("hardware_target", "")
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("event_buffer_size", 10000)

        model_name = self.get_parameter("model").value
        input_type = self.get_parameter("input_type").value
        self.T = self.get_parameter("T").value
        device = self.get_parameter("device").value
        hardware_target = self.get_parameter("hardware_target").value

        # Resolve device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Load SNN Model ---
        self.get_logger().info(f"Loading SNN model: {model_name}")
        try:
            self.model_loader = ModelLoader(
                model_name, device=device, hardware_target=hardware_target
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

        self.get_logger().info(
            f"Model loaded: {self.model_loader.num_params:,} params | "
            f"{self.model_loader.if_count + self.model_loader.lif_count} "
            f"spiking neurons | Accuracy: {self.model_loader.accuracy}%"
        )

        # --- Publishers ---
        self.detection_pub = self.create_publisher(SnnDetection, "/snn/detections", 10)
        self.spike_pub = self.create_publisher(SnnSpikeEvent, "/snn/spikes", 10)
        self.sparsity_pub = self.create_publisher(Float32, "/snn/sparsity", 10)
        self.status_pub = self.create_publisher(SnnStatus, "/snn/status", 10)

        # --- Subscribers ---
        if input_type in ("auto", "image"):
            self.image_sub = self.create_subscription(
                Image, "/camera/image", self.image_callback, 10
            )
            self.get_logger().info("Subscribed to /camera/image")

        if input_type in ("auto", "events") and HAS_EVENT_MSGS:
            self.event_sub = self.create_subscription(
                EventArray, "/dvs/events", self.event_callback, 10
            )
            self.get_logger().info("Subscribed to /dvs/events")
            self.event_buffer = []
            self.event_buffer_max = self.get_parameter("event_buffer_size").value

        # Class names (from model metadata)
        self.class_names = self._get_class_names(model_name)

        self.get_logger().info(
            f"SNN Inference Node ready | Model: {model_name} | Device: {device}"
        )

    def _get_class_names(self, model_name):
        """Get class names based on model."""
        if "mnist" in model_name:
            return [str(i) for i in range(10)]
        elif "nmnist" in model_name:
            return [str(i) for i in range(10)]
        elif "cifar10" in model_name:
            return ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]
        return None

    # ------------------------------------------------------------------
    # Image Callback (regular camera)
    # ------------------------------------------------------------------
    def image_callback(self, msg):
        """Process regular camera image through SNN."""
        try:
            # Convert ROS Image → tensor
            tensor = image_to_tensor(msg)
            tensor = tensor.to(self.model_loader.device)

            # Run SNN inference
            output, spike_stats = self.model_loader.infer_4d(tensor)

            # Publish results
            self._publish_results(output, spike_stats)

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}", throttle_duration_sec=5.0)

    # ------------------------------------------------------------------
    # Event Camera Callback
    # ------------------------------------------------------------------
    def event_callback(self, msg):
        """Buffer events and process when buffer is full."""
        self.event_buffer.extend(msg.events)

        if len(self.event_buffer) >= self.event_buffer_max:
            self._process_event_buffer()

    def _process_event_buffer(self):
        """Convert buffered events to tensor and run SNN."""
        if not self.event_buffer:
            return

        try:
            # Create a pseudo-EventArray for the converter
            class EventArrayMsg:
                def __init__(self, events):
                    self.events = events

            event_msg = EventArrayMsg(self.event_buffer)
            tensor = events_to_tensor(event_msg, T=self.T)
            tensor = tensor.to(self.model_loader.device)

            # Run SNN on temporal event data
            output, spike_stats = self.model_loader.infer_5d(tensor)

            # Publish results
            self._publish_results(output, spike_stats)

            # Clear buffer
            self.event_buffer = []

        except Exception as e:
            self.get_logger().error(f"Event processing error: {e}", throttle_duration_sec=5.0)
            self.event_buffer = []

    # ------------------------------------------------------------------
    # Publish Results
    # ------------------------------------------------------------------
    def _publish_results(self, output, spike_stats):
        """Publish detection, spike events, sparsity, and status."""
        import time

        # 1. Detection — structured SnnDetection message
        det_msg = detection_to_msg(output, self.class_names)
        det_msg.sparsity = spike_stats["sparsity"]
        det_msg.total_spikes = int(spike_stats["total_spikes"])
        det_msg.total_neurons = int(spike_stats["total_activations"])
        self.detection_pub.publish(det_msg)

        # 2. Per-layer spike events — structured SnnSpikeEvent messages
        for spike_msg in spike_stats_to_msg(spike_stats, self.model_loader):
            self.spike_pub.publish(spike_msg)

        # 3. Sparsity — lightweight Float32 for fast monitoring
        sparsity_msg = Float32()
        sparsity_msg.data = spike_stats["sparsity"]
        self.sparsity_pub.publish(sparsity_msg)

        # 4. Status — periodic structured status message
        if spike_stats["total_activations"] > 0:
            status_msg = status_to_msg(self.model_loader, spike_stats, 0.0)
            self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SNNInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down SNN Inference Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
