import gym
import random
import time
import signal
import sys

import numpy as np
import cv2 as cv

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty



def interuppt_handler(signum, frame):
    print("Signal handler!!!")
    sys.exit(-2) #Terminate process here as catching the signal removes the close process behaviour of Ctrl-C



def simulate(env):
    global epsilon, epsilon_decay
    # while not rospy.is_shutdown():
    try:
        for episode in range(1, MAX_EPISODES):
            print("============= STARTING NEW EPISODE ===============")

            state = env.reset()
            total_reward = 0

            for t in range(1, MAX_TRY):
                # In the beginning, do random action to learn
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state])

                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                print("next_state: ", next_state)
                print("total_reward: ", total_reward)
                print("EPISODE:", episode)
                print("TRY:", t)

                if done or t >= MAX_TRY:
                    print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
                    break

                print("STATE: ", state)
                q_value = q_table[state][action]
                best_q = np.max(q_table[next_state])

                # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
                q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

                state = next_state

            # exploring rate decay
            if epsilon >= 0.005:
                epsilon *= epsilon_decay
    except KeyboardInterrupt:
        pass



class GazeboAutoVehicleEnv():
    def __init__(self):
        self.IMAGE_TOPIC = "/vehicle_camera/image_raw"
        self.CMDVEL_TOPIC = "vehicle/cmd_vel"
        self.GZRESET_TOPIC = "/gazebo/reset_world"
        # self.GZPAUSE_TOPIC = '/gazebo/pause_physics'
        # self.GZUNPAUSE_TOPIC = '/gazebo/unpause_physics'

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(np.array([0]),
                                            np.array([20]),
                                            dtype=np.int32)
        rospy.init_node('gym', anonymous=True)
        rospy.Subscriber(self.IMAGE_TOPIC, Image, self.image_callback)
        self.vel_pub = rospy.Publisher(self.CMDVEL_TOPIC, Twist, queue_size=5)
        rospy.wait_for_service(self.GZRESET_TOPIC)
        self.reset_proxy = rospy.ServiceProxy(self.GZRESET_TOPIC, Empty)
        # self.pause = rospy.ServiceProxy(self.GZPAUSE_TOPIC, Empty)
        # self.unpause = rospy.ServiceProxy(self.GZUNPAUSE_TOPIC, Empty)

    def image_callback(self, img):
        self.slope = self._process_image(img)
        pass

    def _process_image(self, img):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(img, "bgr8")
        gray = bridge.imgmsg_to_cv2(img, "mono8")
        _, gray = cv.threshold(gray, 160, 255, cv.THRESH_BINARY);

        H,W = gray.shape
        clipped = gray[int(H*2/3):, :]

        cnt, _, _, centroids = cv.connectedComponentsWithStats(clipped);

        if cnt < 3:
            return None

        dist_left_lane = 0
        dist_right_lane = 0
        left_lane = []
        right_lane = []

        for cent in centroids:
            dist = cent[0] - W / 2
            if dist < dist_left_lane:
                dist_left_lane = dist
                left_lane = cent
            if dist > dist_right_lane:
                dist_right_lane = dist
                right_lane = cent

        # print("LANES:", left_lane, right_lane)

        if left_lane == [] or right_lane == [] or (left_lane == right_lane).all():
            return None

        goal_x = int((left_lane[0] + right_lane[0]) / 2)
        goal_y = int((left_lane[1] + right_lane[1]) / 2)
        error = goal_x - W / 2
        slope = int((error * (180 / W) + 90) / 9)


        clipped = cv.cvtColor(clipped, cv.COLOR_GRAY2RGB)
        cv.circle(clipped,(goal_x, goal_y), 2, (0,255,0), 2)

        cv.imshow("clipped", clipped)
        cv.waitKey(1)
        cv.imwrite("gray.png", gray)
        cv.imwrite("clipped.png", clipped)

        return slope


    def step(self, action):
        self.speed = 0.3
        self.turn = 0.0
        if action == 0:
            self.turn = -0.2
        elif action == 1:
            self.speed = 0.5
        elif action == 2:
            self.turn = 0.2


        twist = Twist()
        twist.linear.x = self.speed
        twist.angular.z = self.turn

        print("-----------------------------")
        print(action)
        print(twist)
        print("-----------------------------")

        # self.unpause()
        self.vel_pub.publish(twist)

        reward = 1
        done = False

        # obs = tuple(self.observation_space.sample())
        # msg = rospy.wait_for_message(self.IMAGE_TOPIC, Image, timeout=5)
        # self.pause()

        # slope = self._process_image(msg)
        slope = self.slope
        obs = slope

        if slope != None:
            reward = 5
        elif slope == None:
            done = True
            reward = -10

        return obs, reward, done, {}

    def reset(self):
        print("======================= RESETTING ==================")

        self.reset_proxy()

        # self.unpause()
        time.sleep(1)
        # msg = rospy.wait_for_message(self.IMAGE_TOPIC, Image, timeout=5)
        # self.pause()

        # slope = self._process_image(msg)
        slope = self.slope
        obs = slope

        return obs



if __name__ == "__main__":
    signal.signal(signal.SIGINT, interuppt_handler)

    env = GazeboAutoVehicleEnv()

    MAX_EPISODES = 9999
    MAX_TRY = 100000
    epsilon = 1
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    q_table = np.zeros(num_box + (env.action_space.n,))
    print("num_box.shape:", np.array(num_box).shape)
    print("q_table.shape:", q_table.shape)
    print(env.observation_space.high, env.observation_space)

    simulate(env)
