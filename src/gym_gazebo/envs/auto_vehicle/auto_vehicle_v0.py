import gym
import rospy
import roslaunch
import time
import numpy as np
import copy
import math
import os
import atexit

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
from std_srvs.srv import Empty




moveBindings = {
    'i': (1,0,0,0),
    'o': (1,0,0,-1),
    'j': (0,0,0,1),
    'l': (0,0,0,-1),
    'u': (1,0,0,1),
    ',': (-1,0,0,0),
    '.': (-1,0,0,1),
    'm': (-1,0,0,-1),
    'O': (1,-1,0,0),
    'I': (1,0,0,0),
    'J': (0,1,0,0),
    'L': (0,-1,0,0),
    'U': (1,1,0,0),
    '<': (-1,0,0,0),
    '>': (-1,-1,0,0),
    'M': (-1,1,0,0),
    't': (0,0,1,0),
    'b': (0,0,-1,0),
}

speedBindings={
    'q': (1.1,1.1),
    'z': (.9,.9),
    'w': (1.1,1),
    'x': (.9,1),
    'e': (1,1.1),
    'c': (1,.9),
}


IMAGE_TOPIC = "/vehicle_camera/image_raw"
CMDVEL_TOPIC = "vehicle/cmd_vel"
GZRESET_TOPIC = "/gazebo/reset_world"




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

def process_image(imgmsg):
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
    cv2.imwrite("canny.png", image)

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

    if lines == []:
        print("lines == []")
        print(lines)
        return None

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
        if line[0] < img_center + thres and line[2] < img_center + thres:
            left_lines.append(line.tolist())
            left_points.append([line[0], line[1]])
            left_points.append([line[2], line[3]])
        elif line[0] > img_center + thres and line[2] > img_center + thres:
            right_lines.append(line.tolist())
            right_points.append([line[0], line[1]])
            right_points.append([line[2], line[3]])

    print("left lines: ", left_lines)
    print("right lines: ", right_lines)
    print()
    print("left points: ", left_points)
    print("right points: ", right_points)
    print()

    if left_lines == [] or right_lines == []:
        return None


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


    print("left lane: ", left_lane)
    print("right lane: ", right_lane)

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
    print("vanishing point: ", vanishing_point)

    slope = (vanishing_point[0] - (width // 2)) / (vanishing_point[1] - height)
    slope = math.degrees(math.atan(slope))
    slope = int((slope + 90) / 9)
    print("SLOPE: ", slope)
    print("-----------------------------")


    return slope


class GazeboAutoVehiclev0Env(gazebo_env.GazeboEnv):
    def __init__(self):
        print("==========================================")
        atexit.register(self.cleanup)

        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboAutoVehicle_v0.launch")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0]),
                                            np.array([20]),
                                            dtype=np.int32)
        # rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback)
        self.vel_pub = rospy.Publisher(CMDVEL_TOPIC, Twist, queue_size=5)
        rospy.wait_for_service(GZRESET_TOPIC)
        self.reset_proxy = rospy.ServiceProxy(GZRESET_TOPIC, Empty, persistent=True)

        self.x = 0
        self.y = 0
        self.z = 0
        self.th = 0.0
        self.speed = 0.3
        self.turn = 0.1

    def cleanup(self):
        self.reset_proxy.close()

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
        key = 'z'
        if action == 0:
            print("action 0")
            key = 'i'
            self.speed = 0.5
            self.turn = 0.4
        elif action == 1:
            print("action 1")
            key = 'u'
            self.speed = 0.8
            self.turn = 0.0
        elif action == 2:
            key = 'o'
            print("action 2")
            self.speed = 0.5
            self.turn = 0.4

        # self.speed = 0.5
        # self.turn = 0.0


        if key in moveBindings.keys():
            self.x = moveBindings[key][0]
            self.y = moveBindings[key][1]
            self.z = moveBindings[key][2]
            self.th = moveBindings[key][3]
        elif key in speedBindings.keys():
            self.speed = self.speed * speedBindings[key][0]
            self.turn  = self.turn  * speedBindings[key][1]

        twist = Twist()
        # Copy state into twist message.
        twist.linear.x = self.x * self.speed
        twist.linear.y = self.y * self.speed
        twist.linear.z = self.z * self.speed
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = self.th * self.turn

        print("-----------------------------")
        print(action, key)
        print(twist)
        print("-----------------------------")

        self.vel_pub.publish(twist)

        reward = 1
        done = False

        # obs = tuple(self.observation_space.sample())
        msg = rospy.wait_for_message(IMAGE_TOPIC, Image, timeout=5)

        slope = process_image(msg)
        obs = slope

        if slope != None and slope > 7 and slope < 14:
            reward = 5
        elif slope == None or slope > 18 or slope < 2:
            print("SLOPE", slope)
            done = True
            reward = -10

        return obs, reward, done, {}

    def reset(self):
        # obs = tuple(self.observation_space.sample())

        print("======================= RESETTING ==================")
        # FIXME: after reset, the received image isn't up to date
        # FIX: use /reset_world instead of /reset_simulation
        self.reset_proxy()
        msg = rospy.wait_for_message(IMAGE_TOPIC, Image, timeout=5)

        slope = process_image(msg)
        obs = slope

        return obs

