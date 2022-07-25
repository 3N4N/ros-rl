import gym
import gym_gazebo
import random
import time
import numpy as np



def simulate():
    global epsilon, epsilon_decay
    for episode in range(1, MAX_EPISODES):

        # Init environment
        state = env.reset()
        total_reward = 0

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

        if episode % 100 == 0:
            # save q_table
            np.save(f"{DIR_CHECKPOINT}/q_table-{str(episode).rjust(5,'0')}", q_table)
            pass


if __name__ == "__main__":
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
