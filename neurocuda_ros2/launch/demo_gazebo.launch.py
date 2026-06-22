#!/usr/bin/env python3
"""
Gazebo simulation — TurtleBot4 + NeuroCUDA SNN inference.

Launches:
  1. TurtleBot4 in Gazebo (with camera)
  2. NeuroCUDA SNN inference node (lifecycle)
  3. Spike visualization (lifecycle)
  4. Lifecycle manager (auto-boots)

Requirements:
  - Ubuntu 24.04 + ROS2 Jazzy
  - sudo apt install ros-jazzy-turtlebot4-simulator ros-jazzy-ros-gz
  - pip install neurocuda

Usage:
  ros2 launch neurocuda_ros2 demo_gazebo.launch.py
  ros2 launch neurocuda_ros2 demo_gazebo.launch.py model:=vgg5_cifar10
  ros2 launch neurocuda_ros2 demo_gazebo.launch.py world:=warehouse

What you see:
  - Gazebo window with TurtleBot4 in a room
  - /snn/detections published as SnnDetection messages
  - Robot camera feed processed through spiking neural network
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode, Node
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression
import os


def generate_launch_description():
    model_arg = DeclareLaunchArgument(
        "model", default_value="neurocuda/vgg5-cifar10-snn",
        description="SNN model for inference",
    )
    world_arg = DeclareLaunchArgument(
        "world", default_value="",
        description="Gazebo world file (empty = default room)",
    )
    use_gazebo_arg = DeclareLaunchArgument(
        "use_gazebo", default_value="true",
        description="Set to false to skip Gazebo (only run SNN)",
    )

    # ---- TurtleBot4 + Gazebo ----
    # Standard TurtleBot4 simulation launch
    turtlebot_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                "/opt/ros/jazzy/share/turtlebot4_ignition_bringup/launch",
                "turtlebot4_ignition.launch.py",
            ])
        ]),
        condition=IfCondition(LaunchConfiguration("use_gazebo")),
    )

    # ---- NeuroCUDA SNN Inference ----
    snn_node = LifecycleNode(
        package="neurocuda_ros2",
        executable="snn_infer",
        name="snn_inference",
        parameters=[{
            "model": LaunchConfiguration("model"),
            "input_type": "image",
            "device": "auto",
            "T": 16,
        }],
        output="screen",
    )

    spike_viz = LifecycleNode(
        package="neurocuda_ros2",
        executable="spike_viz",
        name="spike_viz",
        parameters=[{"model": LaunchConfiguration("model")}],
        output="screen",
    )

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
        model_arg, world_arg, use_gazebo_arg,
        LogInfo(msg=["🤖 NeuroCUDA Gazebo Demo | Model: ", LaunchConfiguration("model")]),
        turtlebot_bringup,
        snn_node, spike_viz, lifecycle_mgr,
        LogInfo(msg=[
            "\n🧠 NeuroCUDA + TurtleBot4 running!",
            "\n  Gazebo: robot driving with camera",
            "\n  SNN: classifying camera feed in real-time",
            "\n",
            "\n  Monitor:",
            "\n    ros2 topic echo /snn/detections",
            "\n    ros2 topic echo /snn/sparsity",
            "\n    ros2 topic echo /snn/status",
        ]),
    ])
