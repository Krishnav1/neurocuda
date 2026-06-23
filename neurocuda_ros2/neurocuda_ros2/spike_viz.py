#!/usr/bin/env python3
"""
NeuroCUDA Spike Visualization — ROS2 Managed Lifecycle Node.

Monitors SNN spike activity and publishes per-layer spike statistics
in real-time. Only active when the parent SNN node is active.

Publishes:
  /snn/spikes (neurocuda_msgs/SnnSpikeEvent) — per-layer spike counts
  /snn/sparsity (std_msgs/Float32) — overall sparsity %

Lifecycle:
  Configure (load model) → Activate (start monitoring) →
  Deactivate (pause) → Cleanup (free)

Usage:
  ros2 run neurocuda_ros2 spike_viz --ros-args -p model:=vgg5_cifar10
"""

import rclpy
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn

from std_msgs.msg import Float32
from neurocuda_msgs.msg import SnnSpikeEvent


class SpikeVizNode(Node):
    """Managed lifecycle node for SNN spike visualization."""

    def __init__(self, node_name="spike_viz"):
        super().__init__(node_name=node_name)
        self.model_loader = None
        self.pub_timer = None
        self._rate = 10.0

    # ==================================================================
    # LIFECYCLE: on_configure
    # ==================================================================
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("⚙️  Configuring spike visualization...")

        try:
            self.declare_parameter("model", "neurocuda/cnn-nmnist-snn")
            self.declare_parameter("publish_rate", 10.0)
            self.declare_parameter("device", "auto")

            model_name = self.get_parameter("model").value
            self._rate = self.get_parameter("publish_rate").value
            device_opt = self.get_parameter("device").value

            import torch
            if device_opt == "auto":
                device_opt = "cuda" if torch.cuda.is_available() else "cpu"

            from neurocuda_ros2.model_loader import ModelLoader
            self.get_logger().info(f"  Loading model: {model_name}")
            self.model_loader = ModelLoader(model_name, device=device_opt)

            self.get_logger().info(
                f"  ✅ Model loaded: "
                f"{self.model_loader.if_count + self.model_loader.lif_count} "
                f"spiking layers"
            )

            # Publishers (regular — lifecycle managed via node state)
            self.spike_pub = self.create_publisher(
                SnnSpikeEvent, "/snn/spikes", 10)
            self.sparsity_pub = self.create_publisher(
                Float32, "/snn/sparsity", 10)
            self.layer_info_pub = self.create_publisher(
                SnnSpikeEvent, "/snn/layer_info", 10)

            self.get_logger().info("  ✅ Configured — ready to activate")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"  ❌ Configure failed: {e}")
            return TransitionCallbackReturn.FAILURE

    # ==================================================================
    # LIFECYCLE: on_activate
    # ==================================================================
    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("▶️  Activating — spike monitoring started")
        self._publish_layer_info()
        self.pub_timer = self.create_timer(1.0 / self._rate, self.publish_spike_data)
        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # LIFECYCLE: on_deactivate
    # ==================================================================
    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("⏸️  Deactivating — pausing spike monitoring")
        if self.pub_timer:
            self.destroy_timer(self.pub_timer)
            self.pub_timer = None
        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # LIFECYCLE: on_cleanup
    # ==================================================================
    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("🧹 Cleaning up...")
        try:
            self.destroy_publisher(self.spike_pub)
            self.destroy_publisher(self.sparsity_pub)
            self.destroy_publisher(self.layer_info_pub)
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
        self.get_logger().info("🛑 Shutting down Spike Viz")
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
    # Publishing
    # ------------------------------------------------------------------
    def _publish_layer_info(self):
        """Publish one SnnSpikeEvent per spiking layer (static info)."""
        from models import IFNeuron, LIFNeuron
        for name, module in self.model_loader.model.named_modules():
            if isinstance(module, (IFNeuron, LIFNeuron)):
                neuron_type = "IF" if isinstance(module, IFNeuron) else "LIF"
                msg = SnnSpikeEvent()
                msg.layer_name = name
                msg.neuron_type = neuron_type
                msg.spike_count = 0
                msg.total_neurons = 0
                msg.sparsity = 0.0
                self.layer_info_pub.publish(msg)

    def publish_spike_data(self):
        """Publish current per-layer spike statistics."""
        if self.model_loader is None:
            return
        stats = self.model_loader._get_spike_stats()

        # Sparsity
        sparsity_msg = Float32()
        sparsity_msg.data = stats["sparsity"]
        self.sparsity_pub.publish(sparsity_msg)

        # Per-layer spike events
        if stats["per_layer"]:
            for name, data in sorted(stats["per_layer"].items()):
                msg = SnnSpikeEvent()
                msg.layer_name = name
                msg.neuron_type = self.model_loader._get_neuron_type(name)
                msg.spike_count = int(data.get("spikes", 0))
                msg.total_neurons = int(data.get("total", 0))
                msg.sparsity = float(data.get("sparsity", 0.0))
                self.spike_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpikeVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
