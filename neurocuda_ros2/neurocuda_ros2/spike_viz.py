#!/usr/bin/env python3
"""
NeuroCUDA Spike Visualization — Monitor SNN activity in real-time.

Publishes spike raster data and sparsity metrics for visualization.
Can be used with RViz2 or Foxglove for live spike monitoring.

Usage:
    ros2 run neurocuda_ros2 spike_viz
    ros2 run neurocuda_ros2 spike_viz --ros-args -p model:=cnn-nmnist-snn

Output topics:
    /snn/spike_raster — per-layer spike counts over time
    /snn/sparsity — overall sparsity percentage
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, String


class SpikeVizNode(Node):
    """Ros2 node for visualizing SNN spike activity."""

    def __init__(self):
        super().__init__("spike_viz")

        self.declare_parameter("model", "neurocuda/cnn-nmnist-snn")
        self.declare_parameter("publish_rate", 10.0)  # Hz
        self.declare_parameter("device", "auto")

        model_name = self.get_parameter("model").value
        rate = self.get_parameter("publish_rate").value

        # Load model
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        from neurocuda_ros2.model_loader import ModelLoader
        self.get_logger().info(f"Loading model for visualization: {model_name}")
        self.model_loader = ModelLoader(model_name, device=device)

        # Publishers
        self.raster_pub = self.create_publisher(Float32MultiArray, "/snn/spike_raster", 10)
        self.sparsity_pub = self.create_publisher(Float32, "/snn/sparsity", 10)
        self.layer_info_pub = self.create_publisher(String, "/snn/layer_info", 10, latch=True)

        # Timer
        self.timer = self.create_timer(1.0 / rate, self.publish_spike_data)

        # Publish static layer info once
        self._publish_layer_info()

        self.get_logger().info(
            f"Spike Viz ready | Model: {model_name} | "
            f"{self.model_loader.if_count + self.model_loader.lif_count} spiking layers"
        )

    def _publish_layer_info(self):
        """Publish static information about spiking layers."""
        info_msg = String()
        layers = []
        for name, module in self.model_loader.model.named_modules():
            from models import IFNeuron, LIFNeuron
            if isinstance(module, (IFNeuron, LIFNeuron)):
                neuron_type = "IF" if isinstance(module, IFNeuron) else "LIF"
                layers.append(f"{name}:{neuron_type}")
        info_msg.data = ",".join(layers)
        self.layer_info_pub.publish(info_msg)

    def publish_spike_data(self):
        """Publish current spike statistics."""
        stats = self.model_loader._get_spike_stats()

        # Sparsity
        sparsity_msg = Float32()
        sparsity_msg.data = stats["sparsity"]
        self.sparsity_pub.publish(sparsity_msg)

        # Per-layer spike counts (raster)
        if stats["per_layer"]:
            raster_data = []
            for name, data in sorted(stats["per_layer"].items()):
                raster_data.append(float(data.get("spikes", 0)))
                raster_data.append(float(data.get("total", 0)))
                raster_data.append(float(data.get("sparsity", 0)))

            raster_msg = Float32MultiArray()
            raster_msg.data = raster_data
            self.raster_pub.publish(raster_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpikeVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Spike Viz")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
