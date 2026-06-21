"""
NeuroCUDA ROS2 — Spiking Neural Network inference and control for ROS2 robots.

Nodes:
    snn_infer    — Event camera / image → SNN → detections
    snn_control  — Robot state → SNN DQN → actions
    spike_viz    — Visualize SNN spike activity in RViz2

Usage:
    ros2 run neurocuda_ros2 snn_infer --ros-args -p model:=cnn-nmnist-snn
    ros2 run neurocuda_ros2 snn_control --ros-args -p model:=dqn-cartpole-snn
    ros2 run neurocuda_ros2 spike_viz

Launch:
    ros2 launch neurocuda_ros2 infer.launch.py model:=cnn-nmnist-snn
    ros2 launch neurocuda_ros2 control.launch.py model:=dqn-cartpole-snn
    ros2 launch neurocuda_ros2 demo_nmnist.launch.py
"""

__version__ = "0.2.0"
