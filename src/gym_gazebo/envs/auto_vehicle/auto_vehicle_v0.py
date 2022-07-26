import gym
import rospy
import roslaunch
import time
import numpy as np
import copy
import math
import os
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from gym.utils import seeding

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState


import cv2
import numpy as np
import rospy
import traceback
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


IMAGE_TOPIC = "/vehicle_camera/image_raw"


def get_isolated_region(image):
    height, width = image.shape
    #isolate the gradients that correspond to the lane lines
    triangle = np.array([[(0, height), (int(width/2), int(height/2)), (width, height)]])
    #create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    #create a mask (triangle that isolates the region of interest in our image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def process_image(image):
    # percent by which the image is resized
    scale_percent = 100

    # calculate the 50 percent of original dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    resized_image = cv2.resize(image, dsize)
    cv2.imwrite("canny_edges_resize.png", resized_image)

    # Make a copy of resized image
    copy = np.copy(resized_image)

    # find the edges
    edges = cv2.Canny(copy, 50, 150)

    # isolate a certain region in the image where the lane lines are
    isolated_region = get_isolated_region(edges)

    # Hough Transform, to find those white pixels from isolated region into actual lines
    # DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array,
    lines = cv2.HoughLinesP(isolated_region, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    print(lines)
    print()

    img_center = width // 2
    thres = 5
    left_lines = []
    right_lines = []
    left_points = []
    right_points = []
    left_lane = [width + thres, -1, -1, -1]
    right_lane = [width + thres, -1, -1, -1]


    for line in lines:
        line = line[0]
        # print(line[0], line[1], line[2], line[3])
        if line[0] < img_center + thres and line[2] < img_center + thres:
            left_lines.append(line.tolist())
            left_points.append([line[0], line[1]])
            left_points.append([line[2], line[3]])
        elif line[0] > img_center + thres and line[2] > img_center + thres:
            right_lines.append(line.tolist())
            right_points.append([line[0], line[1]])
            right_points.append([line[2], line[3]])

    print(left_lines)
    print(right_lines)
    print()
    print(left_points)
    print(right_points)
    print()


    # FIXME: this is too hacky
    # same point can duplicate in a lane
    for point in left_points:
        if point[0] < left_lane[0]:
            left_lane[0] = point[0]
            left_lane[1] = point[1]
        if point[0] > left_lane[2]:
            left_lane[2] = point[0]
            left_lane[3] = point[1]

    for point in right_points:
        if point[0] < right_lane[0]:
            right_lane[0] = point[0]
            right_lane[1] = point[1]
        if point[0] > right_lane[2]:
            right_lane[2] = point[0]
            right_lane[3] = point[1]


    print(left_lane)
    print(right_lane)

    x1 = left_lane[0];
    y1 = left_lane[1];
    x2 = left_lane[2];
    y2 = left_lane[3];

    x3 = right_lane[0];
    y3 = right_lane[1];
    x4 = right_lane[2];
    y4 = right_lane[3];

    vanishing_point = [-1, -1]
    vanishing_point[0] \
        = ((((x1*y2 - y1*x2) * (x3 - x4)) - ((x1 - x2) * (x3*y4 - y3*x4))) \
           / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)))

    vanishing_point[1] \
        = (((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) \
           / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)));

    vanishing_point = [int(i) for i in vanishing_point]
    print(vanishing_point)

    slope = (vanishing_point[0] - (width // 2)) / (vanishing_point[1] - height)
    slope = math.degrees(math.atan(slope))
    slope = int((slope + 90) / 9)
    print(slope, end="\n\n")


    return slope


class GazeboAutoVehiclev0Env(gazebo_env.GazeboEnv):
    def __init__(self):
        print("==========================================")
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboAutoVehicle_v0.launch")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0]),
                                            np.array([20]),
                                            dtype=np.int32)
        rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback)

    def image_callback(*msg):
        print("Received an image!")
        # try:
        #     # Convert your ROS Image message to OpenCV2
        #     bridge = CvBridge()
        #     image = bridge.imgmsg_to_cv2(msg, "bgr8")

        #     # Detect road and get the decision to move forward or turn
        #     road_detection = RoadDetection(image)
        #     decision = road_detection.process_image()

        #     # control vehicle
        #     move_vehicle(decision)

        # except  Exception as e:
        #     # print("exception: ")
        #     print(e)
        #     traceback.print_exc()


    def _seed(self, seed=None):
        # rospy.spin_once()
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # if action == 0:
        #     print("action 0")
        # elif action == 1:
        #     print("action 1")
        # elif action == 2:
        #     print("action 2")

        reward = 1
        done = False

        # obs = tuple(self.observation_space.sample())
        msg = rospy.wait_for_message(IMAGE_TOPIC, Image, timeout=5)

        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, "bgr8")

        slope = process_image(image)
        obs = slope
        return obs, reward, done, {}

    def reset(self):
        # obs = tuple(self.observation_space.sample())
        msg = rospy.wait_for_message(IMAGE_TOPIC, Image, timeout=5)

        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, "bgr8")

        slope = process_image(image)
        obs = slope

        return obs

