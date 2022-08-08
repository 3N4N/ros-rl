# FIXME:
# The car performs well.
# But the q_table doesn't make sense.
# np.argmax(q_table) should be [0, 1, 2, X] (X is don't-care).
# But it's [2, 1, 0, X].


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
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty



def interuppt_handler(signum, frame):
    print("Signal handler!!!")
    sys.exit(-2) #Terminate process here as catching the signal removes the close process behaviour of Ctrl-C



def simulate(env, q_table, alpha, epsilon, epsilon_decay):
    try:
        for episode in range(1, MAX_EPISODES):
            print("============= STARTING NEW EPISODE ===============")

            state = env.reset()
            total_reward = 0

            if state == None:
                episode -= 1
                continue

            done = False

            while not done:
                # In the beginning, do random action to learn
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state])

                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                print("-----------------------------")
                print("EPISODE:", episode)
                print("STATE: ", state)
                print("NEXT STATE: ", next_state)
                print("REWARD: ", reward)
                print("TOTAL REWARD: ", total_reward)
                print("EPSILON:", epsilon)
                print("-----------------------------")

                if next_state == None:
                    next_state = env.observation_space.high

                prev_q = q_table[state][action]
                best_max = np.max(q_table[next_state])

                # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
                q_table[state][action] = (1 - alpha) * prev_q + alpha * (reward + gamma * best_max)

                best_actions = [np.argmax(q_table[state]) for state in range(q_table.shape[0])]
                print("Best actions:", best_actions)

                state = next_state

                if done:
                    print("Episode %d finished with total reward = %f." % (episode, total_reward))
                    break


            if epsilon >= 0.005:
                epsilon *= epsilon_decay

            if episode % 50 == 0:
                np.save("qtable-"+str(episode)+".npy", q_table)

    except KeyboardInterrupt:
        pass



class GazeboAutoVehicleEnv():
    def __init__(self, H, W):
        self.IMAGE_TOPIC = "/vehicle_camera/image_raw"
        self.CMDVEL_TOPIC = "vehicle/cmd_vel"
        self.GZRESET_TOPIC = "/gazebo/reset_world"
        self.GZPAUSE_TOPIC = '/gazebo/pause_physics'
        self.GZUNPAUSE_TOPIC = '/gazebo/unpause_physics'
        self.MODEL_TOPIC = '/gazebo/model_states'

        self.H,self.W = H,W
        self.finished = False

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(np.array([0]),
                                            np.array([3]),
                                            dtype=np.int32)
        rospy.init_node('gym', anonymous=True)
        rospy.Subscriber(self.IMAGE_TOPIC, Image, self.image_callback)
        rospy.Subscriber(self.MODEL_TOPIC, ModelStates, self.modelstate_callback)
        self.vel_pub = rospy.Publisher(self.CMDVEL_TOPIC, Twist, queue_size=5)

        rospy.wait_for_service(self.GZRESET_TOPIC)
        self.reset_proxy = rospy.ServiceProxy(self.GZRESET_TOPIC, Empty)
        self.pause = rospy.ServiceProxy(self.GZPAUSE_TOPIC, Empty)
        self.unpause = rospy.ServiceProxy(self.GZUNPAUSE_TOPIC, Empty)

    def image_callback(self, img):
        self.slope = self._process_image(img)
        pass

    def modelstate_callback(self, states):
        vehicle_pose = states.pose[states.name.index("vehicle")].position
        goal_pose = states.pose[states.name.index("Mailbox")].position
        if vehicle_pose.x > goal_pose.x and vehicle_pose.y > goal_pose.y:
            print("FINISHED!")
            self.finished = True

    def _process_image(self, img):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(img, "bgr8")
        gray = bridge.imgmsg_to_cv2(img, "mono8")
        _, gray = cv.threshold(gray, 160, 255, cv.THRESH_BINARY);

        clipped = gray[int(self.H*2/3):, :]

        cnt, _, _, centroids = cv.connectedComponentsWithStats(clipped);

        if cnt < 3:
            return None

        dist_left_lane = 0
        dist_right_lane = 0
        left_lane = []
        right_lane = []

        for cent in centroids:
            dist = cent[0] - self.W / 2
            if dist < dist_left_lane:
                dist_left_lane = dist
                left_lane = cent
            if dist > dist_right_lane:
                dist_right_lane = dist
                right_lane = cent

        # print("LANES:", left_lane, right_lane)

        if left_lane == [] or right_lane == [] or (left_lane == right_lane).all():
            return None

        if left_lane[0] < 30 or right_lane[0] > self.W - 30:
            return None

        goal_x = int((left_lane[0] + right_lane[0]) / 2)
        goal_y = int((left_lane[1] + right_lane[1]) / 2)
        error = goal_x - self.W / 2

        if error <= -20:
            slope = 0
        elif -20 < error < 20:
            slope = 1
        elif error >= 20:
            slope = 2

        clipped = cv.cvtColor(clipped, cv.COLOR_GRAY2RGB)
        cv.circle(clipped,(goal_x, goal_y), 2, (0,255,0), 2)
        cv.circle(clipped,(int(self.W/2), goal_y), 2, (255,0,0), 2)

        cv.imshow("clipped", clipped)
        cv.waitKey(1)
        cv.imwrite("gray.png", gray)
        cv.imwrite("clipped.png", clipped)

        return slope


    def step(self, action):
        self.speed = 0.3
        self.turn = 0.0
        if action == 0:
            self.turn = 0.2
        elif action == 1:
            self.speed = 0.5
        elif action == 2:
            self.turn = -0.2


        twist = Twist()
        twist.linear.x = self.speed
        twist.angular.z = self.turn

        self.unpause()
        self.vel_pub.publish(twist)
        self.pause()

        slope = self.slope
        obs = slope

        print("-----------------------------")
        print("SLOPE:", slope)
        print("ACTION:", action)
        # print(twist)
        print("-----------------------------")

        if slope != None:
            done = False
            if action == self.prev_slope:
                reward = 10
            else:
                reward = -10
        else:
            done = True
            reward = -10000

        self.prev_slope = slope
        if self.finished:
            done = True

        return obs, reward, done, {}

    def reset(self):
        print("======================= RESETTING ==================")

        self.reset_proxy()
        self.finished = False

        self.unpause()
        time.sleep(1)
        self.pause()

        slope = self.slope
        obs = slope

        self.prev_slope = slope

        return obs



if __name__ == "__main__":
    signal.signal(signal.SIGINT, interuppt_handler)

    env = GazeboAutoVehicleEnv(600, 800)

    MAX_EPISODES = 9999
    epsilon = 1
    epsilon_decay = 0.9
    # epsilon = 0.1
    # epsilon_decay = 1
    alpha = 0.1
    gamma = 0.6
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    q_table = np.zeros(num_box + (env.action_space.n,))
    # q_table = np.zeros([env.observation_space.low, env.action_space.n])
    print("num_box.shape:", np.array(num_box).shape)
    print("q_table.shape:", q_table.shape)
    print("num_box", num_box)
    print(env.observation_space.high, env.observation_space)

    simulate(env, q_table, alpha, epsilon, epsilon_decay)
