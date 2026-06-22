"""NeuroCUDA ROS2 — SNN inference and control for robots."""
from setuptools import setup, find_packages
import os
from glob import glob

package_name = "neurocuda_ros2"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Krishna Varma",
    maintainer_email="krishna@neurocuda.dev",
    description="NeuroCUDA ROS2 — Spiking Neural Network inference and control for ROS2 robots",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "snn_infer = neurocuda_ros2.snn_inference_node:main",
            "snn_control = neurocuda_ros2.snn_control_node:main",
            "spike_viz = neurocuda_ros2.spike_viz:main",
            "lifecycle_mgr = neurocuda_ros2.lifecycle_manager:main",
        ],
    },
)
