#!/usr/bin/env python3
"""
Launch SNN inference node with configurable model and camera.

Usage:
    ros2 launch neurocuda_ros2 infer.launch.py
    ros2 launch neurocuda_ros2 infer.launch.py model:=cnn-nmnist-snn
    ros2 launch neurocuda_ros2 infer.launch.py model:=robotics-perception-snn input_type:=events
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    # Model name
    model_arg = DeclareLaunchArgument(
        "model",
        default_value="neurocuda/cnn-nmnist-snn",
        description="SNN model to use for inference",
    )

    # Input type: auto, image, or events
    input_type_arg = DeclareLaunchArgument(
        "input_type",
        default_value="auto",
        description="Input type: auto, image, or events",
    )

    # Device
    device_arg = DeclareLaunchArgument(
        "device",
        default_value="auto",
        description="Device: auto, cuda, or cpu",
    )

    # Timesteps
    T_arg = DeclareLaunchArgument(
        "T",
        default_value="16",
        description="Number of timesteps for temporal integration",
    )

    # SNN Inference Node
    snn_node = Node(
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

    # Spike Viz Node (optional monitoring)
    spike_viz_node = Node(
        package="neurocuda_ros2",
        executable="spike_viz",
        name="spike_viz",
        parameters=[{
            "model": LaunchConfiguration("model"),
        }],
        output="screen",
    )

    return LaunchDescription([
        model_arg,
        input_type_arg,
        device_arg,
        T_arg,
        LogInfo(msg=LaunchConfiguration("model")),
        snn_node,
        spike_viz_node,
    ])
