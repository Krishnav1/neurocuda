#!/usr/bin/env python3
"""
NeuroCUDA SNN Benchmark Node — ROS2 Managed Lifecycle Node.

Measures and publishes real-time SNN performance metrics:
  - Latency (ms per inference)
  - Throughput (inferences per second)
  - Spike sparsity (per layer and overall)
  - Memory usage (MB)
  - Energy estimate (µJ per inference, modeled)

Publishes:
  /snn/benchmark (neurocuda_msgs/SnnStatus) — per-inference metrics
  /snn/benchmark_summary (std_msgs/String) — periodic summary table

Lifecycle:
  Unconfigured → Configure (load model) → Activate (start measuring) → Deactivate (stop)

Usage:
  ros2 run neurocuda_ros2 benchmark --ros-args -p model:=neurocuda/mlp-mnist-snn
"""

import rclpy
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from lifecycle_msgs.msg import State as LifecycleState

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from neurocuda_msgs.msg import SnnSpikeEvent, SnnStatus

import numpy as np
import torch
import time
import os
from collections import deque

from neurocuda_ros2.model_loader import (
    ModelLoader, image_to_tensor,
    spike_stats_to_msg, status_to_msg,
)

# Try importing neurobench
try:
    from neurocuda.neurobench import NeuroBenchReporter
    HAS_NEUROBENCH = True
except ImportError:
    HAS_NEUROBENCH = False


class BenchmarkNode(Node):
    """Managed lifecycle node for SNN performance benchmarking."""

    def __init__(self, node_name="snn_benchmark"):
        super().__init__(node_name=node_name)
        self.model_loader = None
        self._model_name = None
        self._device_str = "cpu"
        self._hardware_target = ""

        # Metrics storage
        self._latencies = deque(maxlen=1000)
        self._sparsities = deque(maxlen=1000)
        self._inference_count = 0
        self._start_time = None
        self._total_spikes = 0
        self._total_neurons = 0
        self._peak_memory_mb = 0.0

    # ==================================================================
    # LIFECYCLE: on_configure
    # ==================================================================
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("⚙️  Configuring benchmark node...")

        try:
            self.declare_parameter("model", "neurocuda/mlp-mnist-snn")
            self.declare_parameter("device", "auto")
            self.declare_parameter("hardware_target", "")
            self.declare_parameter("T", 16)
            self.declare_parameter("camera_topic", "/camera/image")
            self.declare_parameter("summary_interval_s", 10.0)
            self.declare_parameter("warmup_iterations", 10)

            self._model_name = self.get_parameter("model").value
            self._device_str = self.get_parameter("device").value
            self._hardware_target = self.get_parameter("hardware_target").value
            self.T = self.get_parameter("T").value
            self._camera_topic = self.get_parameter("camera_topic").value
            self._summary_interval = self.get_parameter("summary_interval_s").value
            self._warmup = self.get_parameter("warmup_iterations").value

            if self._device_str == "auto":
                self._device_str = "cuda" if torch.cuda.is_available() else "cpu"

            # Load SNN model
            self.get_logger().info(f"  Model: {self._model_name}")
            self.get_logger().info(f"  Device: {self._device_str}")
            self.model_loader = ModelLoader(
                self._model_name, device=self._device_str,
                hardware_target=self._hardware_target
            )
            self.get_logger().info(
                f"  ✅ Model: {self.model_loader.num_params:,} params | "
                f"{self.model_loader.if_count + self.model_loader.lif_count} spiking neurons"
            )

            # Publishers
            self.benchmark_pub = self.create_publisher(
                SnnStatus, "/snn/benchmark", 10)
            self.summary_pub = self.create_publisher(
                String, "/snn/benchmark_summary", 10)
            self.sparsity_pub = self.create_publisher(
                Float32, "/snn/sparsity", 10)

            # Camera subscription
            self.get_logger().info(f"  Subscribing to: {self._camera_topic}")
            self.image_sub = self.create_subscription(
                Image, self._camera_topic, self.image_callback, 10)

            # Summary timer (created in activate)
            self._summary_timer = None

            self.get_logger().info("  ✅ Benchmark node configured")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"  ❌ Configure failed: {e}")
            return TransitionCallbackReturn.FAILURE

    # ==================================================================
    # LIFECYCLE: on_activate
    # ==================================================================
    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("▶️  Activating — benchmark running")
        if self.model_loader:
            self.model_loader.reset_state()
        self._inference_count = 0
        self._start_time = time.time()
        self._latencies.clear()
        self._sparsities.clear()
        self._total_spikes = 0
        self._total_neurons = 0

        # Summary timer
        self._summary_timer = self.create_timer(
            self._summary_interval, self._publish_summary)

        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # LIFECYCLE: on_deactivate / on_cleanup / on_shutdown
    # ==================================================================
    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("⏸️  Deactivating benchmark")
        if self._summary_timer:
            self.destroy_timer(self._summary_timer)
        # Print final stats before deactivating
        self._publish_summary(final=True)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("🧹 Cleaning up benchmark node")
        try:
            self.destroy_publisher(self.benchmark_pub)
            self.destroy_publisher(self.summary_pub)
            self.destroy_publisher(self.sparsity_pub)
            self.destroy_subscription(self.image_sub)
        except Exception:
            pass
        if self.model_loader:
            del self.model_loader
            self.model_loader = None
            torch.cuda.empty_cache()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("🛑 Shutting down benchmark")
        return self.on_cleanup(state)

    def on_error(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().error("⚠️  Error state — cleaning up")
        return self.on_cleanup(state)

    # ------------------------------------------------------------------
    # Image Callback — measure and benchmark
    # ------------------------------------------------------------------
    def image_callback(self, msg):
        if self.model_loader is None:
            return
        try:
            # Reset IF neuron state for each image
            self.model_loader.reset_state()

            t0 = time.perf_counter()

            # Convert + infer
            tensor = image_to_tensor(msg)
            tensor = tensor.to(self.model_loader.device)
            output, spike_stats = self.model_loader.infer_4d(tensor)

            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            # Track metrics
            self._latencies.append(latency_ms)
            self._sparsities.append(spike_stats["sparsity"])
            self._inference_count += 1
            self._total_spikes += spike_stats["total_spikes"]
            self._total_neurons += spike_stats["total_activations"]

            # Track peak memory (GPU)
            if self._device_str == "cuda":
                mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                if mem_mb > self._peak_memory_mb:
                    self._peak_memory_mb = mem_mb

            # Publish per-inference benchmark
            bench_msg = status_to_msg(
                self.model_loader, spike_stats, inference_time_ms=latency_ms)
            self.benchmark_pub.publish(bench_msg)

            # Publish sparsity
            sp_msg = Float32()
            sp_msg.data = spike_stats["sparsity"]
            self.sparsity_pub.publish(sp_msg)

        except Exception as e:
            self.get_logger().error(
                f"Benchmark inference error: {e}", throttle_duration_sec=5.0)

    # ------------------------------------------------------------------
    # Summary — periodic report
    # ------------------------------------------------------------------
    def _publish_summary(self, final=False):
        """Compute and publish benchmark summary statistics."""
        if self._inference_count < 1:
            return

        now = time.time()
        elapsed = now - self._start_time if self._start_time else 1.0

        latencies = np.array(self._latencies) if self._latencies else np.array([0])
        sparsities = np.array(self._sparsities) if self._sparsities else np.array([0])

        # Compute stats
        avg_lat = float(np.mean(latencies))
        p50_lat = float(np.percentile(latencies, 50))
        p95_lat = float(np.percentile(latencies, 95))
        p99_lat = float(np.percentile(latencies, 99))
        throughput = self._inference_count / elapsed
        avg_sparsity = float(np.mean(sparsities))
        avg_spikes = self._total_spikes / max(self._inference_count, 1)
        total_mem = (self.model_loader.num_params * 4) / (1024 * 1024)  # MB (float32)

        # Energy estimate (modeled, NOT measured on silicon)
        # Using Loihi 2 published constants: 0.9 pJ per spike (synaptic op)
        # and 4.6 pJ per MAC (dense equivalent)
        total_synapses = 0
        for m in self.model_loader.model.modules():
            if isinstance(m, torch.nn.Linear):
                total_synapses += m.in_features * m.out_features
            elif isinstance(m, torch.nn.Conv2d):
                total_synapses += m.in_channels * m.out_channels * m.kernel_size[0]**2

        spike_rate = (100.0 - avg_sparsity) / 100.0
        syn_ops_per_inf = total_synapses * spike_rate * self.model_loader.t
        dense_macs = total_synapses * self.model_loader.t

        # Loihi 2 energy constants (pJ)
        E_SPIKE = 0.9   # pJ per synaptic operation (Loihi 2 published)
        E_MAC = 4.6     # pJ per MAC (45nm CMOS published)
        energy_per_inf_uj = (syn_ops_per_inf * E_SPIKE + dense_macs * 0.05 * E_MAC) / 1e6
        gpu_energy_uj = (dense_macs * E_MAC) / 1e6
        energy_reduction = (1.0 - energy_per_inf_uj / max(gpu_energy_uj, 1e-6)) * 100

        # Build summary string
        header = "BENCHMARK" if not final else "FINAL BENCHMARK"
        summary = (
            f"\n{'='*60}\n"
            f"  {header} — {self._model_name}\n"
            f"  Backend: {self._device_str} | T={self.model_loader.t}\n"
            f"{'='*60}\n"
            f"  Inferences: {self._inference_count} | Elapsed: {elapsed:.1f}s\n"
            f"  Throughput: {throughput:.1f} ips\n"
            f"  Latency:   avg={avg_lat:.2f}ms  p50={p50_lat:.2f}ms  "
            f"p95={p95_lat:.2f}ms  p99={p99_lat:.2f}ms\n"
            f"  Sparsity:  {avg_sparsity:.1f}% (avg {avg_spikes:.0f} spikes/inf "
            f"across {total_synapses:,} synapses)\n"
            f"  Memory:    params={self.model_loader.num_params:,} "
            f"({total_mem:.1f} MB float32)\n"
            f"  Energy (modeled, Loihi 2 constants):\n"
            f"    SNN sparse: {energy_per_inf_uj:.2f} µJ/inference\n"
            f"    GPU dense:  {gpu_energy_uj:.2f} µJ/inference\n"
            f"    Reduction:  {energy_reduction:.0f}%\n"
            f"{'='*60}\n"
        )

        self.get_logger().info(summary)

        # Publish as ROS2 message
        summary_msg = String()
        summary_msg.data = summary
        self.summary_pub.publish(summary_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
