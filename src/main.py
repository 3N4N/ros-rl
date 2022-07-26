import gym
import gym_gazebo
import random
import time
import numpy as np
import rospy
import signal
import sys


def interuppt_handler(signum, frame):
    print("Signal handler!!!")
    sys.exit(-2) #Terminate process here as catching the signal removes the close process behaviour of Ctrl-C



def simulate():
    global epsilon, epsilon_decay
    # while not rospy.is_shutdown():
    try:
        for episode in range(1, MAX_EPISODES):
            print("lsdkfjsldkfjsldkfjsldkfjsldkfjsldkfjsldfkj")

            # Init environment
            state = env.reset()
            total_reward = 0
            print(state)

            # AI tries up to MAX_TRY times
            for t in range(1, MAX_TRY):

                # In the beginning, do random action to learn
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state])

                # Do action and get result
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # Get correspond q value from state, action pair
                q_value = q_table[state][action]
                best_q = np.max(q_table[next_state])

                # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
                q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

                # Set up for the next iteration
                state = next_state

                # Draw games
                env.render()

                # When episode is done, print reward
                if done or t >= MAX_TRY:
                    print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
                    break

            # exploring rate decay
            if epsilon >= 0.005:
                epsilon *= epsilon_decay
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interuppt_handler)
    nh = rospy.init_node('autonomous_vehicle')

    env = gym.make("GazeboAutoVehicle-v0")

    MAX_EPISODES = 9999
    MAX_TRY = 1000
    epsilon = 1
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    print(np.array(num_box).shape)
    # print(env.action_space.n, num_box)
    q_table = np.zeros(num_box + (env.action_space.n,))
    print(q_table.shape)
    print(env.observation_space.high)
    print(env.observation_space)
    simulate()
