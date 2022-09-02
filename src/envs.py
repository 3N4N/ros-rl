import signal
import sys
import time
import gym
import numpy as np
import cv2 as cv



import rospy
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty


from util import _process_image

import gym
import numpy as np


class GazeboAutoVehicleEnv(gym.Env):
# class GazeboAutoVehicleEnv():
    metadata = {
        # "render_modes": ["human", "rgb_array", "single_rgb_array"],
        # "render_fps": 30,
    }

    def __init__(self, H, W):
        self.IMAGE_TOPIC = "/vehicle_camera/image_raw"
        self.CMDVEL_TOPIC = "vehicle/cmd_vel"
        self.GZRESET_TOPIC = "/gazebo/reset_world"
        # self.GZPAUSE_TOPIC = '/gazebo/pause_physics'
        # self.GZUNPAUSE_TOPIC = '/gazebo/unpause_physics'
        self.MODEL_TOPIC = '/gazebo/model_states'

        self.H,self.W = H,W
        self.finished = False

        self.action_space = gym.spaces.Box(np.array([-1]),
                                           np.array([1]),
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(np.array([-1]),
                                                np.array([1]),
                                                dtype=np.float32)
        rospy.init_node('gym', anonymous=True)
        rospy.Subscriber(self.IMAGE_TOPIC, Image, self.image_callback)
        rospy.Subscriber(self.MODEL_TOPIC, ModelStates, self.modelstate_callback)
        self.vel_pub = rospy.Publisher(self.CMDVEL_TOPIC, Twist, queue_size=5)

        rospy.wait_for_service(self.GZRESET_TOPIC)
        self.reset_proxy = rospy.ServiceProxy(self.GZRESET_TOPIC, Empty)
        # self.pause = rospy.ServiceProxy(self.GZPAUSE_TOPIC, Empty)
        # self.unpause = rospy.ServiceProxy(self.GZUNPAUSE_TOPIC, Empty)

        self.state = None

    def image_callback(self, img):
        self.state = _process_image(img, False)
        pass

    def modelstate_callback(self, states):
        vehicle_pose = states.pose[states.name.index("vehicle")].position
        goal_pose = states.pose[states.name.index("Mailbox")].position
        if vehicle_pose.x > goal_pose.x and vehicle_pose.y > goal_pose.y:
            self.finished = True

    def step(self, action):
        action = action[0].item()

        self.speed = 0.5
        self.turn = action

        twist = Twist()
        twist.linear.x = self.speed
        twist.angular.z = self.turn

        # self.unpause()
        self.vel_pub.publish(twist)

        state = self.state
        obs = np.asarray([state], dtype=self.observation_space.dtype)

        if state != None:
            done = False
            reward = 1 - abs(state)
        else:
            done = True
            reward = -1
            state = self.prev_state
            obs = np.asarray([state], dtype=self.observation_space.dtype)

        if self.finished:
            done = True

        print("-----------------------------")
        print("STATE:", state)
        print("ACTION:", action)
        print("REWARD:", reward)
        print("-----------------------------")

        # self.pause()


        return obs, reward, done, {}

    def reset(self):
        print("======================= RESETTING ==================")

        self.reset_proxy()
        self.finished = False
        # self.unpause()
        time.sleep(0.5)
        # self.pause()

        state = None
        while state is None:
            state = self.state
        print("Reset proxy called - reset")

        self.prev_state = self.state
        obs = np.asarray([state], dtype=self.observation_space.dtype)

        return obs

    def render(self, mode):
        pass
