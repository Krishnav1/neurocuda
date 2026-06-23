# NeuroCUDA ROS2 — Complete Environment
# Spiking Neural Network inference for ROS2 robots
#
# Build:  docker build -t neurocuda-ros2 .
# Run:    docker-compose up
#
# Requires ~8GB disk space. Build takes ~15 minutes.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# ============================================================================
# System packages
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg lsb-release wget ca-certificates \
    python3 python3-pip python3-dev \
    build-essential git cmake \
    libegl1 libegl-dev libgles2 libgles-dev \
    libosmesa6 libosmesa6-dev mesa-utils \
    netcat-openbsd iproute2 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# ROS2 Jazzy
# ============================================================================
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | \
    gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" > \
    /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-base \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-topic-tools \
    ros-jazzy-foxglove-bridge \
    ros-jazzy-rosbag2 \
    ros-jazzy-rosbag2-storage-mcap \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Gazebo Harmonic
# ============================================================================
RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg | \
    gpg --dearmor -o /usr/share/keyrings/gazebo-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable noble main" > \
    /etc/apt/sources.list.d/gazebo-stable.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    gz-harmonic \
    libgz-sim8-plugins \
    libgz-rendering8-ogre2 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Python packages
# ============================================================================
RUN pip install --break-system-packages \
    torch torchvision \
    numpy snntorch \
    && pip install --break-system-packages neurocuda || \
    pip install --break-system-packages git+https://github.com/Krishnav1/neurocuda.git

# ============================================================================
# Workspace setup
# ============================================================================
RUN mkdir -p /neurocuda_ws/src

WORKDIR /neurocuda_ws

# Copy workspaces source (mounted at runtime for development)
# For production: COPY . /neurocuda_ws/src/

# Entry: keep container alive
CMD ["bash", "-c", "source /opt/ros/jazzy/setup.bash && tail -f /dev/null"]
