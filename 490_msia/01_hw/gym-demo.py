# Important: The syntax used here is for the latest gym version (0.26.0)
#            If you are using gym versions older than 0.25.0, then the syntax is different:
#            e.g. observation = env.reset()
#                 observation, reward, done, info = env.step(action)
#            ref: https://www.gymlibrary.dev/api/core/#gym.Env.step

import gym


# Question 1: Cartpole-v0
env = gym.make("CartPole-v1")

# Roll out 10 episdoes
# TODO: Update your policy using the collected data
for episode in range(10):
    print(f"CartPole-v0, episode {episode}")
    # Initiate one episode
    observation, info = env.reset()

    obs_history = []
    reward_history = []
    action_history = []

    terminated = False
    truncated = False

    # Roll out one episode
    while (not terminated) and (not truncated):
        action = env.action_space.sample() # Use your policy here
        observation, reward, terminated, truncated, info = env.step(action)

        obs_history.append(observation)
        reward_history.append(reward)
        action_history.append(action)



env.close()


# Question 2: Pong-v0
# TODO: Update your policy using the collected data
env = gym.make("Pong-v0")

# Roll out 10 episdoes
for episode in range(10):
    print(f"Pong-v0, episode {episode}")
    # Initiate one episode
    observation, info = env.reset()

    obs_history = []
    reward_history = []
    action_history = []

    terminated = False
    truncated = False

    # Roll out one episode
    while (not terminated) and (not truncated):
        action = env.action_space.sample() # Use your policy here
        observation, reward, terminated, truncated, info = env.step(action)

        obs_history.append(observation)
        reward_history.append(reward)
        action_history.append(action)

env.close()