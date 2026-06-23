#!/usr/bin/env python3
"""
NeuroCUDA SNN Benchmark Launch — measures latency, throughput, sparsity, energy.

Usage:
  ros2 launch neurocuda_ros2 benchmark.launch.py
  ros2 launch neurocuda_ros2 benchmark.launch.py model:=neurocuda/mlp-mnist-snn camera_topic:=/camera/image
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    model_arg = DeclareLaunchArgument("model",
        default_value="neurocuda/mlp-mnist-snn",
        description="SNN model to benchmark")
    device_arg = DeclareLaunchArgument("device",
        default_value="auto",
        description="Device: auto, cpu, cuda")
    camera_topic_arg = DeclareLaunchArgument("camera_topic",
        default_value="/camera/image",
        description="Camera topic for inference")
    summary_interval_arg = DeclareLaunchArgument("summary_interval_s",
        default_value="10.0",
        description="Seconds between summary reports")

    # Camera sim to feed images
    camera_sim = Node(
        package="neurocuda_ros2",
        executable="camera_sim",
        name="camera_sim",
        parameters=[{
            "pattern": "checkerboard",
            "rate": 30.0,
            "resolution": [28, 28],
        }],
        output="screen",
    )

    # Benchmark node (lifecycle)
    benchmark_node = LifecycleNode(
        package="neurocuda_ros2",
        executable="benchmark",
        name="snn_benchmark",
        parameters=[{
            "model": LaunchConfiguration("model"),
            "device": LaunchConfiguration("device"),
            "camera_topic": LaunchConfiguration("camera_topic"),
            "summary_interval_s": LaunchConfiguration("summary_interval_s"),
            "T": 16,
        }],
        output="screen",
    )

    # Lifecycle manager to auto-boot
    lifecycle_mgr = TimerAction(
        period=5.0,
        actions=[Node(
            package="neurocuda_ros2",
            executable="lifecycle_mgr",
            name="lifecycle_manager_benchmark",
            parameters=[{
                "node_names": ["snn_benchmark"],
                "auto_manage": True,
            }],
            output="screen",
        )],
    )

    return LaunchDescription([
        model_arg, device_arg, camera_topic_arg, summary_interval_arg,
        LogInfo(msg=["🧠 NeuroCUDA SNN Benchmark"]),
        LogInfo(msg=["📊 Latency, Throughput, Sparsity, Energy"]),
        camera_sim,
        TimerAction(period=2.0, actions=[benchmark_node]),
        lifecycle_mgr,
        LogInfo(msg=["✅ Benchmark running — monitor: ros2 topic echo /snn/benchmark_summary"]),
    ])
