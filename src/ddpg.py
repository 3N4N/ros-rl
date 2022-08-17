import signal
import sys
import time
import gym
import numpy as np
import cv2 as cv

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from util import _process_image, interuppt_handler



def interuppt_handler(signum, frame):
    print("Signal handler!!!")
    sys.exit(-2) #Terminate process here as catching the signal removes the close process behaviour of Ctrl-C


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
        self.observation_space = gym.spaces.Box(np.array([-90]),
                                                np.array([90]),
                                                dtype=np.float32)
        rospy.init_node('gym', anonymous=True)
        rospy.Subscriber(self.IMAGE_TOPIC, Image, self.image_callback)
        rospy.Subscriber(self.MODEL_TOPIC, ModelStates, self.modelstate_callback)
        self.vel_pub = rospy.Publisher(self.CMDVEL_TOPIC, Twist, queue_size=5)

        rospy.wait_for_service(self.GZRESET_TOPIC)
        self.reset_proxy = rospy.ServiceProxy(self.GZRESET_TOPIC, Empty)
        # self.pause = rospy.ServiceProxy(self.GZPAUSE_TOPIC, Empty)
        # self.unpause = rospy.ServiceProxy(self.GZUNPAUSE_TOPIC, Empty)

        self.slope = None

    def image_callback(self, img):
        self.slope = _process_image(img)
        pass

    def modelstate_callback(self, states):
        vehicle_pose = states.pose[states.name.index("vehicle")].position
        goal_pose = states.pose[states.name.index("Mailbox")].position
        if vehicle_pose.x > goal_pose.x and vehicle_pose.y > goal_pose.y:
            print("FINISHED!")
            self.finished = True


    def step(self, action):
        print(type(action[0]))
        self.speed = 0.5

        self.turn = action[0].item()
        # if action == 0:
        #     self.turn = 0.2
        # elif action == 1:
        #     self.speed = 0.9
        # elif action == 2:
        #     self.turn = -0.2


        twist = Twist()
        twist.linear.x = self.speed
        twist.angular.z = self.turn

        # self.unpause()
        self.vel_pub.publish(twist)
        # self.pause()

        slope = self.slope
        obs = np.asarray([slope], dtype=self.observation_space.dtype)

        print("-----------------------------")
        print("SLOPE:", slope)
        print("ACTION:", action[0].item())
        # print(twist)
        print("-----------------------------")

        if slope != None:
            done = False
            reward = 90 - abs(slope)
        else:
            done = True
            reward = -10000

        if self.finished:
            done = True

        if slope == None:
            obs = env.observation_space.high

        return obs, reward, done, {}

    def reset(self):
        print("======================= RESETTING ==================")

        self.reset_proxy()
        self.finished = False
        # print("Reset proxy called - reset")
        # self.unpause()
        # time.sleep(0.5)
        # self.pause()

        slope = None
        while slope is None:
            slope = self.slope
        # print("Slope found! - reset")
        obs = np.asarray([slope], dtype=self.observation_space.dtype)

        if obs == None:
            obs = self.observation_space.high
        # print("Returning - reset")
        return obs




if __name__ == "__main__":
    signal.signal(signal.SIGINT, interuppt_handler)

    env = GazeboAutoVehicleEnv(600, 800)
    check_env(env)

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, buffer_size = 100)
    model.learn(total_timesteps=10000, log_interval=10)

    model_name = "model_ddpg"
    model.save(model_name)
    env = model.get_env()
    del model # remove to demonstrate saving and loading

    model = DDPG.load(model_name)

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
