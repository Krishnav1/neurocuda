#!/usr/bin/env python3
"""
Full NMNIST demo — event camera → SNN → detection → visualization.

Usage:
    ros2 launch neurocuda_ros2 demo_nmnist.launch.py
    ros2 launch neurocuda_ros2 demo_nmnist.launch.py model:=robotics-perception-snn

Requires:
    - NeuroCUDA installed (pip install neurocuda)
    - event_camera_msgs (for event camera replay)
    - Optional: metavision_driver (for real Prophesee camera)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression


def generate_launch_description():
    model_arg = DeclareLaunchArgument(
        "model", default_value="neurocuda/cnn-nmnist-snn",
        description="SNN model for event camera inference",
    )
    use_sim_arg = DeclareLaunchArgument(
        "use_simulation", default_value="true",
        description="Use simulated events instead of real camera",
    )

    # SNN Inference node
    snn_node = Node(
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

    # Spike visualization
    spike_viz = Node(
        package="neurocuda_ros2",
        executable="spike_viz",
        name="spike_viz",
        parameters=[{"model": LaunchConfiguration("model")}],
        output="screen",
    )

    return LaunchDescription([
        model_arg,
        use_sim_arg,
        LogInfo(msg=["Starting NMNIST demo with model: ", LaunchConfiguration("model")]),
        snn_node,
        spike_viz,
        LogInfo(msg=[
            "NeuroCUDA NMNIST Demo running!",
            "\n  Model: ", LaunchConfiguration("model"),
            "\n  Topics:",
            "\n    /snn/detections — classification results",
            "\n    /snn/sparsity — spike sparsity percentage",
            "\n    /snn/spike_raster — per-layer spike counts",
            "\n    /snn/status — model status and metrics",
        ]),
    ])
