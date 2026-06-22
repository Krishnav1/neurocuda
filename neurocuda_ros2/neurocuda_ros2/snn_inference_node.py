#!/usr/bin/env python3
"""
NeuroCUDA SNN Inference Node — ROS2 Managed Lifecycle Node.

Subscribes to:
  /camera/image (sensor_msgs/Image) — regular camera
  /dvs/events (event_camera_msgs/EventArray) — event camera

Publishes:
  /snn/detections (neurocuda_msgs/SnnDetection) — classification results
  /snn/spikes (neurocuda_msgs/SnnSpikeEvent) — per-layer spike activity
  /snn/sparsity (std_msgs/Float32) — overall sparsity %
  /snn/status (neurocuda_msgs/SnnStatus) — model metrics

Lifecycle states:
  Unconfigured → (on_configure)  → Inactive  [load model, allocate GPU]
  Inactive     → (on_activate)   → Active    [start processing]
  Active       → (on_deactivate) → Inactive  [pause, reset state]
  Inactive     → (on_cleanup)    → Unconfigured [free GPU]
  Any          → (on_shutdown)   → Finalized [clean exit]

Usage:
  ros2 launch neurocuda_ros2 infer.launch.py model:=vgg5_cifar10
"""

import rclpy
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from lifecycle_msgs.msg import State as LifecycleState

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

# Event camera messages (optional)
try:
    from event_camera_msgs.msg import EventArray
    HAS_EVENT_MSGS = True
except ImportError:
    HAS_EVENT_MSGS = False


class SNNInferenceNode(Node):
    """Managed lifecycle node for spiking neural network inference."""

    def __init__(self, node_name="snn_inference"):
        super().__init__(node_name=node_name)
        self.model_loader = None
        self.class_names = None
        self.T = 16
        self.event_buffer = []
        self._model_name = None
        self._input_type = "auto"
        self._device_str = "cpu"
        self._hardware_target = ""

    # ==================================================================
    # LIFECYCLE: on_configure
    # Called when node transitions from Unconfigured → Inactive.
    # Load model, create pubs/subs. GPU allocation happens here.
    # ==================================================================
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("⚙️  Configuring — loading SNN model...")

        try:
            # --- Declare parameters ---
            self.declare_parameter("model", "neurocuda/cnn-nmnist-snn")
            self.declare_parameter("input_type", "auto")
            self.declare_parameter("T", 16)
            self.declare_parameter("device", "auto")
            self.declare_parameter("hardware_target", "")
            self.declare_parameter("publish_debug_image", True)
            self.declare_parameter("event_buffer_size", 10000)

            self._model_name = self.get_parameter("model").value
            self._input_type = self.get_parameter("input_type").value
            self.T = self.get_parameter("T").value
            self._device_str = self.get_parameter("device").value
            self._hardware_target = self.get_parameter("hardware_target").value
            event_buffer_max = self.get_parameter("event_buffer_size").value

            # Resolve device
            if self._device_str == "auto":
                import torch
                self._device_str = "cuda" if torch.cuda.is_available() else "cpu"

            # --- Load SNN model ---
            self.get_logger().info(f"  Model: {self._model_name}")
            self.get_logger().info(f"  Device: {self._device_str}")
            self.model_loader = ModelLoader(
                self._model_name, device=self._device_str,
                hardware_target=self._hardware_target
            )

            self.get_logger().info(
                f"  ✅ Model loaded: {self.model_loader.num_params:,} params | "
                f"{self.model_loader.if_count + self.model_loader.lif_count} "
                f"spiking neurons | Accuracy: {self.model_loader.accuracy}%"
            )

            # --- Class names ---
            self.class_names = self._get_class_names(self._model_name)

            # --- Lifecycle publishers (only active when node is Active) ---
            self.detection_pub = self.create_lifecycle_publisher(
                SnnDetection, "/snn/detections", 10)
            self.spike_pub = self.create_lifecycle_publisher(
                SnnSpikeEvent, "/snn/spikes", 10)
            self.sparsity_pub = self.create_lifecycle_publisher(
                Float32, "/snn/sparsity", 10)
            self.status_pub = self.create_lifecycle_publisher(
                SnnStatus, "/snn/status", 10)

            # --- Subscriptions (only active when node is Active) ---
            if self._input_type in ("auto", "image"):
                self.image_sub = self.create_subscription(
                    Image, "/camera/image", self.image_callback, 10)

            if self._input_type in ("auto", "events") and HAS_EVENT_MSGS:
                self.event_sub = self.create_subscription(
                    EventArray, "/dvs/events", self.event_callback, 10)
                self.event_buffer = []
                self.event_buffer_max = event_buffer_max

            self.get_logger().info("  ✅ Configured — ready to activate")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"  ❌ Configure failed: {e}")
            return TransitionCallbackReturn.FAILURE

    # ==================================================================
    # LIFECYCLE: on_activate
    # Called when transitioning Inactive → Active.
    # Reset neuron state, enable processing.
    # ==================================================================
    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("▶️  Activating — SNN inference running")
        if self.model_loader:
            self.model_loader.reset_state()
        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # LIFECYCLE: on_deactivate
    # Called when transitioning Active → Inactive.
    # Reset neuron state, stop accepting new inference.
    # ==================================================================
    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("⏸️  Deactivating — pausing SNN inference")
        if self.model_loader:
            self.model_loader.reset_state()
        self.event_buffer = []
        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # LIFECYCLE: on_cleanup
    # Called when transitioning Inactive → Unconfigured.
    # Free GPU memory, destroy publishers and subscribers.
    # ==================================================================
    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("🧹 Cleaning up — freeing GPU memory...")

        try:
            self.destroy_publisher(self.detection_pub)
            self.destroy_publisher(self.spike_pub)
            self.destroy_publisher(self.sparsity_pub)
            self.destroy_publisher(self.status_pub)
        except Exception:
            pass

        try:
            self.destroy_subscription(self.image_sub)
        except Exception:
            pass

        try:
            if HAS_EVENT_MSGS:
                self.destroy_subscription(self.event_sub)
        except Exception:
            pass

        # Free GPU memory
        if self.model_loader:
            del self.model_loader
            self.model_loader = None
            import torch
            torch.cuda.empty_cache()

        self.get_logger().info("  ✅ Cleaned up — GPU memory freed")
        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # LIFECYCLE: on_shutdown
    # Called when transitioning to Finalized.
    # Last chance cleanup before destruction.
    # ==================================================================
    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("🛑 Shutting down SNN Inference Node")
        self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # LIFECYCLE: on_error
    # Called when an error occurs in Active state.
    # Attempt recovery by transitioning to Unconfigured.
    # ==================================================================
    def on_error(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().error("⚠️  Error state entered — attempting recovery")
        self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_class_names(self, model_name):
        if model_name is None:
            return None
        if "mnist" in model_name:
            return [str(i) for i in range(10)]
        elif "nmnist" in model_name:
            return [str(i) for i in range(10)]
        elif "cifar10" in model_name:
            return ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]
        return None

    # ------------------------------------------------------------------
    # Image Callback
    # ------------------------------------------------------------------
    def image_callback(self, msg):
        if self.model_loader is None:
            return
        try:
            tensor = image_to_tensor(msg)
            tensor = tensor.to(self.model_loader.device)
            output, spike_stats = self.model_loader.infer_4d(tensor)
            self._publish_results(output, spike_stats)
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}", throttle_duration_sec=5.0)

    # ------------------------------------------------------------------
    # Event Camera Callback
    # ------------------------------------------------------------------
    def event_callback(self, msg):
        if self.model_loader is None:
            return
        self.event_buffer.extend(msg.events)
        if len(self.event_buffer) >= self.event_buffer_max:
            self._process_event_buffer()

    def _process_event_buffer(self):
        if not self.event_buffer:
            return
        try:
            class EventArrayMsg:
                def __init__(self, events):
                    self.events = events
            event_msg = EventArrayMsg(self.event_buffer)
            tensor = events_to_tensor(event_msg, T=self.T)
            tensor = tensor.to(self.model_loader.device)
            output, spike_stats = self.model_loader.infer_5d(tensor)
            self._publish_results(output, spike_stats)
            self.event_buffer = []
        except Exception as e:
            self.get_logger().error(f"Event processing error: {e}", throttle_duration_sec=5.0)
            self.event_buffer = []

    # ------------------------------------------------------------------
    # Publish Results
    # ------------------------------------------------------------------
    def _publish_results(self, output, spike_stats):
        # Detection
        det_msg = detection_to_msg(output, self.class_names)
        det_msg.sparsity = spike_stats["sparsity"]
        det_msg.total_spikes = int(spike_stats["total_spikes"])
        det_msg.total_neurons = int(spike_stats["total_activations"])
        self.detection_pub.publish(det_msg)

        # Per-layer spikes
        for spike_msg in spike_stats_to_msg(spike_stats, self.model_loader):
            self.spike_pub.publish(spike_msg)

        # Sparsity
        sparsity_msg = Float32()
        sparsity_msg.data = spike_stats["sparsity"]
        self.sparsity_pub.publish(sparsity_msg)

        # Status
        if spike_stats["total_activations"] > 0:
            status_msg = status_to_msg(self.model_loader, spike_stats, 0.0)
            self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SNNInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
