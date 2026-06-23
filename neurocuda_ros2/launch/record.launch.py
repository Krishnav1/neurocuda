#!/usr/bin/env python3
"""
Rosbag Record Launch — records robot sensor data for reproducible testing.

Records:
  - /camera/image (sensor_msgs/Image)
  - /snn/detections (neurocuda_msgs/SnnDetection)
  - /snn/spikes (neurocuda_msgs/SnnSpikeEvent)
  - /snn/sparsity (std_msgs/Float32)
  - /snn/status (neurocuda_msgs/SnnStatus)
  - /snn/benchmark (neurocuda_msgs/SnnStatus)
  - /snn/benchmark_summary (std_msgs/String)

Output: .mcap file (ROS2 Jazzy default format)

Usage:
  ros2 launch neurocuda_ros2 record.launch.py
  ros2 launch neurocuda_ros2 record.launch.py bag_name:=warehouse_run_1
  ros2 launch neurocuda_ros2 record.launch.py topics:=[/camera/image,/snn/detections]
  ros2 launch neurocuda_ros2 record.launch.py max_duration_s:=60
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, LogInfo, ExecuteProcess,
    TimerAction, Shutdown
)
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    bag_name_arg = DeclareLaunchArgument("bag_name",
        default_value="neurocuda_session",
        description="Name for the rosbag file")
    topics_arg = DeclareLaunchArgument("topics",
        default_value="[/camera/image,/snn/detections,/snn/spikes,/snn/sparsity,/snn/status,/snn/benchmark,/snn/benchmark_summary,/snn/q_values]",
        description="Topics to record (YAML list)")
    max_dur_arg = DeclareLaunchArgument("max_duration_s",
        default_value="0",
        description="Max recording duration in seconds (0 = unlimited)")
    compress_arg = DeclareLaunchArgument("compress",
        default_value="false",
        description="Compress with zstd")
    bag_dir_arg = DeclareLaunchArgument("bag_dir",
        default_value="/neurocuda_ws/bags",
        description="Directory to save bag files")

    bag_path = LaunchConfiguration("bag_dir")
    bag_name = LaunchConfiguration("bag_name")

    record_cmd = ExecuteProcess(
        cmd=["ros2", "bag", "record",
             "-o", bag_name,
             "-a",  # Record all topics
        ],
        output="screen",
        name="rosbag_record",
    )

    return LaunchDescription([
        bag_name_arg, topics_arg, max_dur_arg, compress_arg, bag_dir_arg,
        LogInfo(msg=["🎥 Recording rosbag..."]),
        LogInfo(msg=["📁 Bag name: ", bag_name]),
        LogInfo(msg=["📡 Topics: ALL (use ros2 bag record -t to filter)"]),
        LogInfo(msg=["⏹️  Stop: Ctrl+C or 'ros2 service call /rosbag_record/kill'"]),
        record_cmd,
    ])
