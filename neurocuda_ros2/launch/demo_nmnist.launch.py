#!/usr/bin/env python3
"""
Full NMNIST demo with lifecycle management — event camera → SNN → detection.

Usage:
    ros2 launch neurocuda_ros2 demo_nmnist.launch.py
    ros2 launch neurocuda_ros2 demo_nmnist.launch.py model:=vgg5_cifar10

For real Prophesee/iniVation cameras, set use_simulation:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    model_arg = DeclareLaunchArgument(
        "model", default_value="neurocuda/cnn-nmnist-snn",
        description="SNN model for event camera inference",
    )
    use_sim_arg = DeclareLaunchArgument(
        "use_simulation", default_value="true",
        description="Use simulated events instead of real camera",
    )

    # SNN Inference — lifecycle node
    snn_node = LifecycleNode(
        package="neurocuda_ros2",
        executable="snn_infer",
        name="snn_inference",
        parameters=[{
            "model": LaunchConfiguration("model"),
            "input_type": "events",
            "T": 16,
        }],
        output="screen",
    )

    # Spike visualization — lifecycle node
    spike_viz = LifecycleNode(
        package="neurocuda_ros2",
        executable="spike_viz",
        name="spike_viz",
        parameters=[{"model": LaunchConfiguration("model")}],
        output="screen",
    )

    # Lifecycle manager — boots all nodes automatically
    lifecycle_mgr = Node(
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
        model_arg, use_sim_arg,
        LogInfo(msg=["🧠 NMNIST Demo | Model: ", LaunchConfiguration("model")]),
        snn_node, spike_viz, lifecycle_mgr,
        LogInfo(msg=[
            "NeuroCUDA NMNIST Demo running!",
            "\n  /snn/detections — SnnDetection (class, confidence, top-k)",
            "\n  /snn/spikes — SnnSpikeEvent (per-layer spike activity)",
            "\n  /snn/sparsity — Float32 (overall sparsity %)",
            "\n  /snn/status — SnnStatus (model metrics and device info)",
        ]),
    ])
