#!/usr/bin/env python3
"""
Synthetic camera demo — test NeuroCUDA SNN without any hardware.

Launches:
  1. camera_sim — publishes synthetic test images (CIFAR-10-like patterns)
  2. snn_inference — lifecyle SNN node (classifies images)
  3. spike_viz — per-layer spike monitoring
  4. lifecycle_manager — auto-boots all lifecycle nodes

Works in Docker, WSL2, cloud VM — no GPU, camera, or Gazebo needed.

Usage:
  ros2 launch neurocuda_ros2 demo_synthetic.launch.py
  ros2 launch neurocuda_ros2 demo_synthetic.launch.py pattern:=airplane rate:=5.0
  ros2 launch neurocuda_ros2 demo_synthetic.launch.py model:=vgg5_cifar10

Then monitor:
  ros2 topic echo /snn/detections
  ros2 topic echo /snn/sparsity
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    model_arg = DeclareLaunchArgument(
        "model", default_value="neurocuda/cnn-nmnist-snn",
        description="SNN model from NeuroCUDA hub",
    )
    pattern_arg = DeclareLaunchArgument(
        "pattern", default_value="random",
        description="Test pattern: random, airplane, car, bird, cat, dog, ship, checkerboard, gradient, circle",
    )
    rate_arg = DeclareLaunchArgument(
        "rate", default_value="2.0",
        description="Image publish rate (Hz)",
    )
    resolution_arg = DeclareLaunchArgument(
        "resolution", default_value="[64, 64]",
        description="Image resolution [H, W]",
    )

    # Camera simulator — standard node (not lifecycle, it's simple)
    camera_node = Node(
        package="neurocuda_ros2",
        executable="camera_sim",
        name="camera_sim",
        parameters=[{
            "pattern": LaunchConfiguration("pattern"),
            "rate": LaunchConfiguration("rate"),
            "resolution": LaunchConfiguration("resolution"),
        }],
        output="screen",
    )

    # SNN Inference — lifecycle node
    snn_node = LifecycleNode(
        package="neurocuda_ros2",
        executable="snn_infer",
        name="snn_inference",
        parameters=[{
            "model": LaunchConfiguration("model"),
            "input_type": "image",
            "device": "cpu",
            "T": 16,
        }],
        output="screen",
    )

    # Spike viz — lifecycle node
    viz_node = LifecycleNode(
        package="neurocuda_ros2",
        executable="spike_viz",
        name="spike_viz",
        parameters=[{"model": LaunchConfiguration("model")}],
        output="screen",
    )

    # Lifecycle manager — boots SNN nodes
    mgr_node = Node(
        package="neurocuda_ros2",
        executable="lifecycle_mgr",
        name="lifecycle_manager_snn",
        parameters=[{
            "node_names": ["snn_inference", "spike_viz"],
            "auto_manage": True,
        }],
        output="screen",
    )

    return LaunchDescription([
        model_arg, pattern_arg, rate_arg, resolution_arg,
        LogInfo(msg=["🧠 NeuroCUDA Synthetic Demo | Model: ", LaunchConfiguration("model")]),
        LogInfo(msg=["📷 Pattern: ", LaunchConfiguration("pattern")]),
        camera_node, snn_node, viz_node, mgr_node,
        LogInfo(msg=[
            "\n🚀 Demo running!",
            "\n  ros2 topic echo /snn/detections  ← see what the SNN detects",
            "\n  ros2 topic echo /snn/sparsity    ← see spike sparsity",
            "\n  ros2 topic list                   ← see all topics",
        ]),
    ])
