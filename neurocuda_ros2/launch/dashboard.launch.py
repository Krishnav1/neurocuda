#!/usr/bin/env python3
"""
NeuroCUDA Foxglove Dashboard Launch — real-time SNN brain visualization.

Starts:
  1. SNN inference node (lifecycle)
  2. Foxglove WebSocket bridge (port 8765)
  3. Lifecycle manager (auto-boot)
  4. Optional: camera_sim or Gazebo feed

Connect: Open Foxglove Studio → "Open Connection" → ws://localhost:8765
         Or use Foxglove web: https://app.foxglove.dev/

Usage:
  ros2 launch neurocuda_ros2 dashboard.launch.py
  ros2 launch neurocuda_ros2 dashboard.launch.py model:=neurocuda/mlp-mnist-snn
  ros2 launch neurocuda_ros2 dashboard.launch.py port:=8765 use_sim_camera:=true
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, LogInfo,
    IncludeLaunchDescription, TimerAction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode, Node
from launch.conditions import IfCondition


def generate_launch_description():
    model_arg = DeclareLaunchArgument("model",
        default_value="neurocuda/mlp-mnist-snn",
        description="SNN model")
    device_arg = DeclareLaunchArgument("device",
        default_value="auto",
        description="Device: auto, cpu, cuda")
    port_arg = DeclareLaunchArgument("port",
        default_value="8765",
        description="Foxglove WebSocket port")
    use_sim_arg = DeclareLaunchArgument("use_sim_camera",
        default_value="true",
        description="Start camera_sim for test images")
    camera_topic_arg = DeclareLaunchArgument("camera_topic",
        default_value="/camera/image",
        description="Camera topic")

    # SNN inference node
    snn_node = LifecycleNode(
        package="neurocuda_ros2",
        executable="snn_infer",
        name="snn_inference",
        parameters=[{
            "model": LaunchConfiguration("model"),
            "device": LaunchConfiguration("device"),
            "input_type": "image",
            "T": 16,
            "camera_topic": LaunchConfiguration("camera_topic"),
        }],
        output="screen",
    )

    # Camera simulator (optional)
    camera_sim = Node(
        package="neurocuda_ros2",
        executable="camera_sim",
        name="camera_sim",
        parameters=[{
            "pattern": "checkerboard",
            "rate": 10.0,
            "resolution": [28, 28],
        }],
        condition=IfCondition(LaunchConfiguration("use_sim_camera")),
        output="screen",
    )

    # Foxglove WebSocket bridge
    foxglove_bridge = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        parameters=[{
            "port": LaunchConfiguration("port"),
            "address": "0.0.0.0",
            "send_buffer_limit": 10000000,
            "use_compression": True,
            "max_update_ms": 50,  # 20Hz update rate
        }],
        output="screen",
    )

    # Lifecycle manager
    lifecycle_mgr = TimerAction(
        period=4.0,
        actions=[Node(
            package="neurocuda_ros2",
            executable="lifecycle_mgr",
            name="lifecycle_manager_dash",
            parameters=[{
                "node_names": ["snn_inference"],
                "auto_manage": True,
            }],
            output="screen",
        )],
    )

    return LaunchDescription([
        model_arg, device_arg, port_arg, use_sim_arg, camera_topic_arg,
        LogInfo(msg=["📊 NeuroCUDA Foxglove Dashboard"]),
        LogInfo(msg=["🌐 Connect: Foxglove Studio → ws://localhost:", LaunchConfiguration("port")]),
        LogInfo(msg=["📷 Camera: ", LaunchConfiguration("camera_topic")]),
        LogInfo(msg=["🧠 Model: ", LaunchConfiguration("model")]),
        LogInfo(msg=[""]),
        LogInfo(msg=["Dashboard panels:"]),
        LogInfo(msg=["  📷 Camera view — /camera/image"]),
        LogInfo(msg=["  🏷️  Detections — /snn/detections"]),
        LogInfo(msg=["  📊 Sparsity — /snn/sparsity"]),
        LogInfo(msg=["  ⚡ Spikes — /snn/spikes"]),
        LogInfo(msg=["  📈 Status — /snn/status"]),
        camera_sim,
        snn_node,
        foxglove_bridge,
        lifecycle_mgr,
        LogInfo(msg=[""]),
        LogInfo(msg=["✅ Dashboard ready! Open Foxglove Studio → ws://localhost:8765"]),
    ])
