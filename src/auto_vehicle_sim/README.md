## Autonomous Vehicle Simulator

#### Introduction

This is a Differential Drive Robot in Gazebo.

#### Environment Setup

Installing **ROS Noetic** at *Ubuntu 20.04.3 LTS*

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

sudo apt install curl # if you haven't already installed curl

curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

sudo apt update
sudo apt install ros-noetic-desktop-full

source /opt/ros/noetic/setup.bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt install python3-rosdep

sudo rosdep init
rosdep update
Now check

$ printenv | grep ROS
#### Dependencies

First, if you installed Gazebo from debians, make sure you've installed the Gazebo development files. If you installed Gazebo from source, you can ignore this step. If you have a release other than gazebo11, replace 11 with whatever version number you have.

$ sudo apt-get install libgazebo11-dev
$ sudo apt-get install libopencv-dev python3-opencv
$ sudo apt-get install ros-noetic-cv-bridge
$ sudo apt-get install python3-numpy
$ sudo apt-get install python3-matplotlib
#### Installing and running the project

To install this plugin project, git clone the repo

$ git clone https://github.com/DSInnovators/ROS-lab.git
$ cd ROS-lab/ROS1/catkin_ws
$ catkin_make
$ source devel/setup.sh
$ roslaunch autonomous_driving_track_world run.launch
You will see like this.
![img.png](img.png)

In seperate terminal, drive the vehicle.


For Python application,

$ rosrun autonomous_vehicle automate_vehicle.py

For C++ application,

$ rosrun auto_vehicle automate_vehicle
#### Output

Vehicle will run on the road.
