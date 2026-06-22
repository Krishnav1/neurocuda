#!/usr/bin/env python3
"""
Launch SNN inference with lifecycle management.

Launches 3 nodes:
  1. snn_inference — lifecycle node (camera → SNN → detections)
  2. spike_viz — lifecycle node (per-layer spike monitoring)
  3. lifecycle_manager_snn — auto-boots lifecycle nodes

Usage:
    ros2 launch neurocuda_ros2 infer.launch.py
    ros2 launch neurocuda_ros2 infer.launch.py model:=vgg5_cifar10 device:=cuda
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessStart
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    model_arg = DeclareLaunchArgument(
        "model", default_value="neurocuda/cnn-nmnist-snn",
        description="SNN model name from NeuroCUDA hub",
    )
    input_type_arg = DeclareLaunchArgument(
        "input_type", default_value="auto",
        description="Input type: auto, image, or events",
    )
    device_arg = DeclareLaunchArgument(
        "device", default_value="auto",
        description="Device: auto, cuda, or cpu",
    )
    T_arg = DeclareLaunchArgument(
        "T", default_value="16",
        description="Timesteps for temporal integration",
    )
    nodes_arg = DeclareLaunchArgument(
        "managed_nodes",
        default_value="['snn_inference', 'spike_viz']",
        description="List of lifecycle nodes to manage",
    )

    # SNN Inference — managed lifecycle node
    snn_node = LifecycleNode(
        package="neurocuda_ros2",
        executable="snn_infer",
        name="snn_inference",
        parameters=[{
            "model": LaunchConfiguration("model"),
            "input_type": LaunchConfiguration("input_type"),
            "device": LaunchConfiguration("device"),
            "T": LaunchConfiguration("T"),
        }],
        output="screen",
    )

    # Spike Visualization — managed lifecycle node
    spike_viz_node = LifecycleNode(
        package="neurocuda_ros2",
        executable="spike_viz",
        name="spike_viz",
        parameters=[{
            "model": LaunchConfiguration("model"),
        }],
        output="screen",
    )

    # Lifecycle Manager — auto-boots all SNN nodes
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
        model_arg, input_type_arg, device_arg, T_arg, nodes_arg,
        LogInfo(msg=["NeuroCUDA SNN Inference | Model: ", LaunchConfiguration("model")]),
        snn_node,
        spike_viz_node,
        lifecycle_mgr,
        LogInfo(msg=[
            "\n🧠 NeuroCUDA SNN Inference running!",
            "\n  Topics:",
            "\n    /snn/detections — SnnDetection (class, confidence, top-k, sparsity)",
            "\n    /snn/spikes — SnnSpikeEvent (per-layer spike counts)",
            "\n    /snn/sparsity — Float32 (overall sparsity %)",
            "\n    /snn/status — SnnStatus (model, accuracy, device, latency)",
        ]),
    ])
