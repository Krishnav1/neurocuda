#!/usr/bin/env python3
"""
Rosbag Replay Launch — plays recorded sensor data into the SNN pipeline.

Pipeline:
  .mcap file → [ros2 bag play] → /camera/image → SNN → /snn/detections

This enables:
  - Reproducible testing (same input every time)
  - Compare ANN vs SNN on identical data
  - Benchmark without needing Gazebo running
  - Share results with reviewers (send the bag file)

Usage:
  ros2 launch neurocuda_ros2 replay.launch.py bag_file:=/path/to/bag
  ros2 launch neurocuda_ros2 replay.launch.py bag_file:=warehouse_run \
      model:=neurocuda/resnet18-cifar10-snn
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, LogInfo, ExecuteProcess,
    TimerAction
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    bag_file_arg = DeclareLaunchArgument("bag_file",
        default_value="",
        description="Path to .mcap rosbag file (required)")
    model_arg = DeclareLaunchArgument("model",
        default_value="neurocuda/mlp-mnist-snn",
        description="SNN model")
    device_arg = DeclareLaunchArgument("device",
        default_value="auto",
        description="Device: auto, cpu, cuda")
    rate_arg = DeclareLaunchArgument("playback_rate",
        default_value="1.0",
        description="Playback speed (1.0 = real-time, 2.0 = 2x)")
    loop_arg = DeclareLaunchArgument("loop",
        default_value="false",
        description="Loop playback")
    benchmark_arg = DeclareLaunchArgument("benchmark",
        default_value="true",
        description="Also run benchmark node")

    # SNN inference node (lifecycle)
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

    # Rosbag playback
    bag_play = ExecuteProcess(
        cmd=["ros2", "bag", "play",
             LaunchConfiguration("bag_file"),
             "--rate", LaunchConfiguration("playback_rate"),
        ],
        output="screen",
        name="rosbag_play",
    )

    # Lifecycle manager
    lifecycle_mgr = TimerAction(
        period=5.0,
        actions=[Node(
            package="neurocuda_ros2",
            executable="lifecycle_mgr",
            name="lifecycle_manager_replay",
            parameters=[{
                "node_names": ["snn_inference"],
                "auto_manage": True,
            }],
            output="screen",
        )],
    )

    actions = [
        bag_file_arg, model_arg, device_arg, rate_arg, loop_arg, benchmark_arg,
        LogInfo(msg=["🔄 Rosbag Replay → SNN Pipeline"]),
        LogInfo(msg=["📁 Bag: ", LaunchConfiguration("bag_file")]),
        LogInfo(msg=["🧠 Model: ", LaunchConfiguration("model")]),
        LogInfo(msg=["📷 Camera → SNN → /snn/detections"]),
        snn_node,
        lifecycle_mgr,
        TimerAction(period=3.0, actions=[bag_play]),
        LogInfo(msg=["✅ Replay running — ros2 topic echo /snn/detections"]),
        LogInfo(msg=["📊 Monitor: ros2 topic hz /snn/detections"]),
    ]

    return LaunchDescription(actions)
