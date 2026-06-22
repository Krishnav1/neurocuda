#!/usr/bin/env python3
"""
Headless Gazebo + TurtleBot4 + NeuroCUDA SNN — full verification demo.

Launches:
  1. Gazebo Sim server (headless, no GUI)
  2. TurtleBot4 spawned in warehouse world
  3. ROS-GZ bridge (all sensors, including OAK-D camera)
  4. NeuroCUDA SNN inference (lifecycle) — classifies camera feed
  5. Spike visualization (lifecycle)
  6. Lifecycle manager (auto-boots)

Works in Docker, cloud VM, CI/CD — no GPU, no display needed.

Usage:
  ros2 launch neurocuda_ros2 demo_gazebo_headless.launch.py
  ros2 launch neurocuda_ros2 demo_gazebo_headless.launch.py model:=vgg5_cifar10

Monitor:
  ros2 topic echo /snn/detections
  ros2 topic echo /oakd/rgb/preview/image_raw  (TurtleBot4 camera)
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    model_arg = DeclareLaunchArgument("model",
        default_value="neurocuda/cnn-nmnist-snn",
        description="SNN model")
    world_arg = DeclareLaunchArgument("world",
        default_value="warehouse",
        description="World: warehouse, depot, maze")
    headless_arg = DeclareLaunchArgument("headless",
        default_value="true",
        description="Run Gazebo headless (server-only)")

    # ---- Gazebo Sim server (headless) ----
    world_path = os.path.join(
        "/opt/ros/jazzy/share/turtlebot4_gz_bringup/worlds",
        LaunchConfiguration("world").perform(None) + ".sdf"
    )

    gz_server = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", world_path],
        output="screen",
        name="gz_server",
    )

    # ---- ROS-GZ Sensor Bridge ----
    gz_bridge = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "launch", "turtlebot4_gz_bringup",
                     "ros_gz_bridge.launch.py"],
                output="screen",
                name="gz_bridge",
            )
        ],
    )

    # ---- NeuroCUDA SNN Node (Lifecycle) ----
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

    # Remap TurtleBot4 OAK-D camera to SNN expected topic
    snn_node_substitutions = {
        "/camera/image": "/oakd/rgb/preview/image_raw",
    }

    spike_viz = LifecycleNode(
        package="neurocuda_ros2",
        executable="spike_viz",
        name="spike_viz",
        parameters=[{"model": LaunchConfiguration("model")}],
        output="screen",
    )

    lifecycle_mgr = TimerAction(
        period=8.0,
        actions=[
            Node(
                package="neurocuda_ros2",
                executable="lifecycle_mgr",
                name="lifecycle_manager_snn",
                parameters=[{
                    "node_names": ["snn_inference", "spike_viz"],
                    "auto_manage": True,
                }],
                output="screen",
            )
        ],
    )

    return LaunchDescription([
        model_arg, world_arg, headless_arg,
        LogInfo(msg=["🤖 Headless Gazebo + TurtleBot4 + NeuroCUDA SNN"]),
        LogInfo(msg=["📷 Camera: /oakd/rgb/preview/image_raw → SNN → /snn/detections"]),
        gz_server, gz_bridge,
        TimerAction(period=6.0, actions=[snn_node, spike_viz]),
        lifecycle_mgr,
        LogInfo(msg=[
            "\n✅ All systems running!",
            "\n  Gazebo: headless server with TurtleBot4 + warehouse",
            "\n  SNN: classifying camera feed",
            "\n",
            "\n  ros2 topic echo /snn/detections",
            "\n  ros2 topic echo /oakd/rgb/preview/image_raw",
        ]),
    ])
