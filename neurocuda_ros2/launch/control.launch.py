#!/usr/bin/env python3
"""
Launch SNN control node.

Usage:
    ros2 launch neurocuda_ros2 control.launch.py
    ros2 launch neurocuda_ros2 control.launch.py model:=dqn-cartpole-snn
    ros2 launch neurocuda_ros2 control.launch.py action_mode:=discrete num_actions:=4
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_arg = DeclareLaunchArgument(
        "model",
        default_value="neurocuda/dqn-cartpole-snn",
        description="SNN control model",
    )
    action_mode_arg = DeclareLaunchArgument(
        "action_mode", default_value="discrete",
        description="Action mode: discrete or continuous",
    )
    num_actions_arg = DeclareLaunchArgument(
        "num_actions", default_value="2",
        description="Number of discrete actions",
    )
    period_arg = DeclareLaunchArgument(
        "publish_period", default_value="0.05",
        description="Control update period (seconds)",
    )

    control_node = Node(
        package="neurocuda_ros2",
        executable="snn_control",
        name="snn_control",
        parameters=[{
            "model": LaunchConfiguration("model"),
            "action_mode": LaunchConfiguration("action_mode"),
            "num_actions": LaunchConfiguration("num_actions"),
            "publish_period": LaunchConfiguration("publish_period"),
        }],
        output="screen",
    )

    return LaunchDescription([
        model_arg, action_mode_arg, num_actions_arg, period_arg,
        control_node,
    ])
