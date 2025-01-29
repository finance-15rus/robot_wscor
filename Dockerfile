FROM osrf/ros:noetic-desktop-full

RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-noetic-moveit \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-joint-state-publisher-gui \
    ros-noetic-ros-controllers \
    ros-noetic-joint-state-controller \
    ros-noetic-position-controllers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /catkin_ws
CMD ["/bin/bash"]