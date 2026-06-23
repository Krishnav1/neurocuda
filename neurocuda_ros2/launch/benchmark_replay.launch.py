#!/usr/bin/env python3
"""
Benchmark Replay Launch — reproducible SNN benchmarking against recorded data.

This is the KEY paper tool:
  1. Record a scene ONCE (ros2 bag record)
  2. Replay it MANY TIMES against different models/configs
  3. Compare results apples-to-apples (identical input)

Usage:
  ros2 launch neurocuda_ros2 benchmark_replay.launch.py bag_file:=/path/to/bag
  ros2 launch neurocuda_ros2 benchmark_replay.launch.py bag_file:=warehouse \
      model:=neurocuda/resnet18-cifar10-snn device:=cuda

Output:
  /snn/benchmark (SnnStatus) — per-inference metrics
  /snn/benchmark_summary (String) — periodic report with:
    - Latency (avg, p50, p95, p99)
    - Throughput (ips)
    - Spike sparsity (%)
    - Energy estimate (µJ)
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, LogInfo, ExecuteProcess, TimerAction
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch.conditions import LaunchConfigurationEquals


def generate_launch_description():
    bag_file_arg = DeclareLaunchArgument("bag_file",
        default_value="",
        description="Path to .mcap rosbag file (required)")
    model_arg = DeclareLaunchArgument("model",
        default_value="neurocuda/mlp-mnist-snn",
        description="SNN model to benchmark")
    device_arg = DeclareLaunchArgument("device",
        default_value="auto",
        description="Device: auto, cpu, cuda")
    playback_rate_arg = DeclareLaunchArgument("playback_rate",
        default_value="1.0",
        description="Playback speed (1.0 = real-time)")
    summary_interval_arg = DeclareLaunchArgument("summary_interval_s",
        default_value="15.0",
        description="Seconds between benchmark summary reports")
    bag_dir_arg = DeclareLaunchArgument("bag_dir",
        default_value="/neurocuda_ws/bags",
        description="Default directory for bag files")

    bag_file = LaunchConfiguration("bag_file")

    # SNN Inference node
    snn_node = LifecycleNode(
        package="neurocuda_ros2",
        executable="snn_infer",
        name="snn_inference",
        parameters=[{
            "model": LaunchConfiguration("model"),
            "device": LaunchConfiguration("device"),
            "input_type": "image",
            "T": 16,
            "camera_topic": "/camera/image",
        }],
        output="screen",
    )

    # Benchmark node
    benchmark_node = LifecycleNode(
        package="neurocuda_ros2",
        executable="benchmark",
        name="snn_benchmark",
        parameters=[{
            "model": LaunchConfiguration("model"),
            "device": LaunchConfiguration("device"),
            "camera_topic": "/camera/image",
            "summary_interval_s": LaunchConfiguration("summary_interval_s"),
            "T": 16,
        }],
        output="screen",
    )

    # Lifecycle manager (boots both nodes)
    lifecycle_mgr = TimerAction(
        period=5.0,
        actions=[Node(
            package="neurocuda_ros2",
            executable="lifecycle_mgr",
            name="lifecycle_manager_benchmark",
            parameters=[{
                "node_names": ["snn_inference", "snn_benchmark"],
                "auto_manage": True,
            }],
            output="screen",
        )],
    )

    # Rosbag playback
    bag_play = ExecuteProcess(
        cmd=["ros2", "bag", "play", bag_file,
             "--rate", LaunchConfiguration("playback_rate"),
        ],
        output="screen",
        name="rosbag_play",
    )

    return LaunchDescription([
        bag_file_arg, model_arg, device_arg, playback_rate_arg,
        summary_interval_arg, bag_dir_arg,
        LogInfo(msg=["🧠 NeuroCUDA Benchmark Replay"]),
        LogInfo(msg=["📁 Bag: ", bag_file]),
        LogInfo(msg=["🧠 Model: ", LaunchConfiguration("model")]),
        LogInfo(msg=["📊 Output: /snn/benchmark_summary"]),
        LogInfo(msg=[""]),
        LogInfo(msg=["Pipeline: .mcap → /camera/image → SNN → /snn/detections"]),
        LogInfo(msg=["          simultaneous → Benchmark → /snn/benchmark_summary"]),
        snn_node,
        benchmark_node,
        lifecycle_mgr,
        TimerAction(period=8.0, actions=[bag_play]),
        LogInfo(msg=[""]),
        LogInfo(msg=["✅ Benchmark replay running!"]),
        LogInfo(msg=["📊 Monitor: ros2 topic echo /snn/benchmark_summary"]),
        LogInfo(msg=["📈 Compare: ros2 topic hz /snn/detections"]),
    ])
