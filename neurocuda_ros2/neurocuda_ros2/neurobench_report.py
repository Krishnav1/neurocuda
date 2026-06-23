#!/usr/bin/env python3
"""
NeuroBench Standard Report Generator for NeuroCUDA ROS2.

Generates a NeuroBench-compliant benchmark report (Nature Comms 2025 format)
from ROS2 SNN pipeline measurements.

Output: Paper-ready markdown table + JSON for NeuroBench submission.

Usage:
  ros2 run neurocuda_ros2 neurobench_report --ros-args \
    -p model:=neurocuda/mlp-mnist-snn -p backend:=cpu
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import time
import json
import os
from datetime import datetime
from collections import deque


class NeuroBenchReportNode(Node):
    """Collects SNN metrics and generates NeuroBench-standard reports."""

    def __init__(self):
        super().__init__("neurobench_report")

        self.declare_parameter("model", "neurocuda/mlp-mnist-snn")
        self.declare_parameter("backend", "cpu")
        self.declare_parameter("dataset", "MNIST")
        self.declare_parameter("T", 16)
        self.declare_parameter("report_interval_s", 30.0)

        self._model_name = self.get_parameter("model").value
        self._backend = self.get_parameter("backend").value
        self._dataset = self.get_parameter("dataset").value
        self.T = self.get_parameter("T").value
        self._interval = self.get_parameter("report_interval_s").value

        # Metric storage
        self.latencies = deque(maxlen=10000)
        self.sparsities = deque(maxlen=10000)
        self.total_spikes = 0
        self.total_neurons = 0
        self.inference_count = 0
        self.start_time = time.time()

        # Load model for metadata
        from neurocuda_ros2.model_loader import ModelLoader, image_to_tensor
        self._loader = ModelLoader(self._model_name, device="cpu")
        self.image_to_tensor = image_to_tensor

        # Count synapses
        import torch
        self.total_synapses = 0
        self.total_params = self._loader.num_params
        for m in self._loader.model.modules():
            if isinstance(m, torch.nn.Linear):
                self.total_synapses += m.in_features * m.out_features
            elif isinstance(m, torch.nn.Conv2d):
                self.total_synapses += m.in_channels * m.out_channels * m.kernel_size[0]**2

        # Subscribe to camera and detection topics
        from sensor_msgs.msg import Image
        from neurocuda_msgs.msg import SnnDetection
        self.camera_sub = self.create_subscription(
            Image, "/camera/image", self.camera_callback, 10)
        self.det_sub = self.create_subscription(
            SnnDetection, "/snn/detections", self.detection_callback, 10)

        # Publisher
        self.report_pub = self.create_publisher(String, "/snn/neurobench_report", 10)

        # Timer for periodic reports
        self.report_timer = self.create_timer(self._interval, self.generate_report)

        self.get_logger().info(
            f"NeuroBench reporter ready | Model: {self._model_name} | "
            f"Backend: {self._backend} | Dataset: {self._dataset}"
        )

    def camera_callback(self, msg):
        """Measure inference latency."""
        t0 = time.perf_counter()
        tensor = self.image_to_tensor(msg)
        self._loader.reset_state()
        with __import__('torch').no_grad():
            _ = self._loader.model(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latencies.append(latency_ms)
        self.inference_count += 1

    def detection_callback(self, msg):
        """Track sparsity and spike counts."""
        self.sparsities.append(msg.sparsity)
        self.total_spikes += int(msg.total_spikes)
        self.total_neurons += int(msg.total_neurons)

    def generate_report(self):
        """Generate a NeuroBench-standard report."""
        if self.inference_count < 10:
            return

        elapsed = time.time() - self.start_time
        lats = np.array(self.latencies)
        spars = np.array(self.sparsities) if self.sparsities else np.array([0])

        # Algorithm metrics
        accuracy = self._loader.accuracy
        avg_sparsity = float(np.mean(spars))
        spike_rate = (100.0 - avg_sparsity) / 100.0
        syn_ops = int(self.total_synapses * spike_rate * self.T)
        effective_macs = int(self.total_synapses * self.T)
        model_size_mb = (self.total_params * 4) / (1024 * 1024)

        # System metrics
        avg_lat = float(np.mean(lats))
        p50_lat = float(np.percentile(lats, 50))
        p95_lat = float(np.percentile(lats, 95))
        throughput = self.inference_count / max(elapsed, 0.001)

        # Energy (Loihi 2 constants)
        E_SPIKE = 0.9   # pJ per synaptic operation
        E_MAC = 50.0    # pJ per MAC (GPU 7nm)
        energy_uj = (syn_ops * E_SPIKE) / 1e6
        dense_energy_uj = (effective_macs * E_MAC) / 1e6
        energy_ratio = dense_energy_uj / max(energy_uj, 1e-6)
        reduction_pct = (1.0 - energy_uj / max(dense_energy_uj, 1e-6)) * 100

        # NeuroBench report
        report = {
            "benchmark": "NeuroBench v2.0",
            "model": self._model_name,
            "backend": self._backend,
            "dataset": self._dataset,
            "date": datetime.now().isoformat(),
            "algorithm": {
                "accuracy_top1": float(accuracy) if isinstance(accuracy, (int, float)) else 0.0,
                "activation_sparsity": round(avg_sparsity / 100, 4),
                "synaptic_operations": syn_ops,
                "effective_macs": effective_macs,
                "timesteps": self.T,
                "parameters": self.total_params,
                "model_size_mb": round(model_size_mb, 2),
            },
            "system": {
                "latency_ms_avg": round(avg_lat, 3),
                "latency_ms_p50": round(p50_lat, 3),
                "latency_ms_p95": round(p95_lat, 3),
                "throughput_ips": round(throughput, 2),
                "energy_uj_per_inference": round(energy_uj, 3),
                "dense_equivalent_energy_uj": round(dense_energy_uj, 3),
                "energy_reduction_pct": round(reduction_pct, 1),
                "energy_vs_dense_ratio": round(energy_ratio, 1),
            },
            "hardware": {
                "type": "simulator",
                "description": "PyTorch CPU SNN inference" if self._backend == "cpu"
                             else "Loihi 2 simulator (8-bit weights, 24-bit V)",
            },
            "energy_constants": {
                "E_SPIKE_pJ": E_SPIKE,
                "E_MAC_pJ": E_MAC,
                "source": "Intel Loihi 2 Technical Brief (2024); 7nm CMOS published",
            },
        }

        # Markdown table
        table = (
            f"\n{'='*70}\n"
            f"  NEUROBENCH STANDARD REPORT\n"
            f"  Model: {self._model_name} | Backend: {self._backend}\n"
            f"  Dataset: {self._dataset} | T={self.T}\n"
            f"  Samples: {self.inference_count} | Elapsed: {elapsed:.0f}s\n"
            f"{'='*70}\n\n"
            f"  Algorithm Track:\n"
            f"    Accuracy:     {accuracy}%\n"
            f"    Sparsity:     {avg_sparsity:.1f}%\n"
            f"    SynOps:       {syn_ops:,}\n"
            f"    Parameters:   {self.total_params:,}\n"
            f"    Model size:   {model_size_mb:.1f} MB\n\n"
            f"  System Track:\n"
            f"    Latency:      avg={avg_lat:.2f}ms p50={p50_lat:.2f}ms p95={p95_lat:.2f}ms\n"
            f"    Throughput:   {throughput:.1f} ips\n"
            f"    Energy (SNN): {energy_uj:.2f} µJ/inference\n"
            f"    Energy (dense):{dense_energy_uj:.2f} µJ/inference\n"
            f"    Reduction:    {reduction_pct:.0f}% ({energy_ratio:.0f}× less)\n\n"
            f"  Comparison Table:\n"
            f"  | Backend | Type | Acc | Lat(ms) | Energy(µJ) | vs Dense |\n"
            f"  |---------|------|-----|---------|------------|----------|\n"
            f"  | {self._backend.upper():<7} | {report['hardware']['type']:<4} | "
            f"{accuracy}% | {avg_lat:.1f} | {energy_uj:.1f} | "
            f"{energy_ratio:.0f}× ({reduction_pct:.0f}%) |\n"
            f"{'='*70}\n"
        )

        self.get_logger().info(table)

        # Publish report
        msg = String()
        msg.data = json.dumps(report, indent=2)
        self.report_pub.publish(msg)

        # Save to file
        os.makedirs("/neurocuda_ws/reports", exist_ok=True)
        report_path = f"/neurocuda_ws/reports/neurobench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        self.get_logger().info(f"Report saved: {report_path}")


def main(args=None):
    rclpy.init(args=args)
    node = NeuroBenchReportNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
