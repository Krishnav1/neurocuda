#!/bin/bash
# NeuroCUDA ROS2 entrypoint — sources ROS2 environment before running commands
source /opt/ros/jazzy/setup.bash
exec "$@"
