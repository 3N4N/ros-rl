import gym
import random
import time
import signal

import numpy as np

import rospy
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

import numpy as np
import matplotlib.pyplot as plt

from util import process_image, interuppt_handler


def simulate(env, q_table, alpha, epsilon, epsilon_decay):
    rewards = []
    avgrewards = []
    try:
        for episode in range(1, MAX_EPISODES):
            print("============= STARTING NEW EPISODE ===============")

            state = env.reset()
            total_reward = 0
            total_steps = 0

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
                # print("REWARDS:", rewards)
                print("AVG REWARDS:", avgrewards)
                print("-----------------------------")

                if next_state == None:
                    next_state = env.observation_space.high

                # Q(s, a) <- (1 - α) * Q(s, a) + α * (r + γ + max(Q(s', a)))
                q_table[state][action] = (1 - alpha) * q_table[state][action] \
                                       + alpha * (reward + gamma * np.max(q_table[next_state]))

                best_actions = [np.argmax(q_table[state]) for state in range(q_table.shape[0])]
                print("Best actions:", best_actions)

                state = next_state
                total_steps += 1

            print("Episode %d finished with total reward = %f." % (episode, total_reward))
            rewards.append(total_reward/total_steps)
            avgrewards.append(np.mean(rewards[-50:]))

            if epsilon >= 0.005:
                epsilon *= epsilon_decay

            # if episode % 10 == 0:
            #     np.save("qtable-"+str(episode)+".npy", q_table)
            #     np.save("reward-"+str(episode)+".npy", np.array(rewards))

        plt.plot(np.array(avgrewards))
        plt.xlabel("No. of Episodes")
        plt.ylabel("Avg. Rewards")
        plt.show()

    except KeyboardInterrupt:
        pass



class GazeboAutoVehicleEnv():
    def __init__(self, H, W):
        self.IMAGE_TOPIC = "/vehicle_camera/image_raw"
        self.CMDVEL_TOPIC = "vehicle/cmd_vel"
        self.GZRESET_TOPIC = "/gazebo/reset_world"
        # self.GZPAUSE_TOPIC = '/gazebo/pause_physics'
        # self.GZUNPAUSE_TOPIC = '/gazebo/unpause_physics'
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
        # self.pause = rospy.ServiceProxy(self.GZPAUSE_TOPIC, Empty)
        # self.unpause = rospy.ServiceProxy(self.GZUNPAUSE_TOPIC, Empty)

    def image_callback(self, img):
        state = process_image(img, True)
        if state == None:
            self.state = None
            return

        error = state * 90 * (self.W / 180)

        if error <= -20:
            self.state = 0
        elif -20 < error < 20:
            self.state = 1
        elif error >= 20:
            self.state = 2

    def modelstate_callback(self, states):
        vehicle_pose = states.pose[states.name.index("vehicle")].position
        goal_pose = states.pose[states.name.index("Mailbox")].position
        if vehicle_pose.x > goal_pose.x and vehicle_pose.y > goal_pose.y:
            print("FINISHED!")
            self.finished = True


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

        # self.unpause()
        self.vel_pub.publish(twist)
        # self.pause()

        state = self.state
        obs = state

        print("-----------------------------")
        print("STATE:", state)
        print("ACTION:", action)
        print("-----------------------------")

        if state != None:
            done = False
            if action == self.prev_state:
                reward = 1
            else:
                reward = -1
        else:
            done = True
            reward = -1e2

        self.prev_state = state
        if self.finished:
            done = True

        return obs, reward, done, {}

    def reset(self):
        print("======================= RESETTING ==================")

        self.reset_proxy()
        self.finished = False

        # self.unpause()
        time.sleep(1)
        # self.pause()

        state = None
        while state is None:
            state = self.state

        obs = state
        self.prev_state = state

        return obs



if __name__ == "__main__":
    signal.signal(signal.SIGINT, interuppt_handler)

    env = GazeboAutoVehicleEnv(600, 800)

    MAX_EPISODES = 100
    epsilon = 1
    epsilon_decay = 0.9
    # epsilon = 0.1
    # epsilon_decay = 1
    alpha = 0.1
    gamma = 0.6
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    q_table = np.zeros(num_box + (env.action_space.n,))
    print("num_box.shape:", np.array(num_box).shape)
    print("q_table.shape:", q_table.shape)
    print("num_box", num_box)
    print(env.observation_space.high, env.observation_space)

    simulate(env, q_table, alpha, epsilon, epsilon_decay)
