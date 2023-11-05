import gym
import warnings
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
import os
import tensorflow as tf
from collections import deque
from math import log

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def check_gpu_availability(use_gpus=None):
    # List all GPUs available to TensorFlow
    gpus = tf.config.list_physical_devices('GPU')

    # Check if any GPUs are available
    if len(gpus) > 0:
        print("GPUs available:", gpus)
        if use_gpus is not None:
            # Set TensorFlow to only use the specified GPUs
            try:
                # Specify which GPUs to use
                tf.config.set_visible_devices([gpus[i] for i in use_gpus], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
    else:
        print("GPU is not available, using CPU instead.")


# Preprocess Mrs. Packman Frames
mspacman_color = 210 + 164 + 74
def preprocess_observation(obs):
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.sum(axis=2)  # to greyscale
    img[img==mspacman_color] = 0  # Improve contrast
    # img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    img = (img / 255.0).astype(np.float32)  # Normalize between 0 and 1
    return img.reshape(88, 80, 1)

# Select agent's action
def select_action(model, state, epsilon, num_actions):
    if random.random() < epsilon:
        return random.choice(range(num_actions)), None  # No Q-value in random selection
    q_values = model(tf.expand_dims(state, axis=0))[0]
    # print(q_values)
    return tf.argmax(q_values).numpy(), q_values.numpy()

# Plot model rewards over time
def plot_rewards(episode_rewards, window_size=100):
    plt.figure(figsize=(10, 6))
    
    # Plot histogram instead of a line chart
    plt.hist(episode_rewards, bins=20, alpha=0.7, label='Episode rewards')
    
    plt.xlabel('Reward Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Calculate mean and standard deviation
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    # Create artifacts folder if it doesn't exist
    artifacts_path = os.path.join(script_dir, 'artifacts')
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)
    
    # Write mean and standard deviation to a text file
    stats_file_path = os.path.join(artifacts_path, 'MsPacman_500_Rollout_Stats.txt')
    with open(stats_file_path, 'w') as f:
        f.write(f"Mean Reward: {mean_reward}\n")
        f.write(f"Standard Deviation: {std_reward}\n")
    
    plt.savefig(os.path.join(artifacts_path, 'MsPacman_500_Rollout_Progress.png'))
    plt.close()

# Train agent to play MsPacman
def MsPacman_RL(model_path, rollout_len=500):
    # Initialize environment and model
    env = gym.make(
        "MsPacman-v0", 
        # render_mode="human"
        )
    trained_model = tf.keras.models.load_model(model_path)
    num_actions = 9
    actual_episode_rewards = []
    epsilon = 0

    # Check model
    trained_model.summary()

    # Training loop
    for episode in range(1, rollout_len+1):
        observation, info = env.reset()
        processed_observation = preprocess_observation(observation)
        actual_reward = []
        terminated = False
        truncated = False

        # Episode loop
        while not terminated and not truncated:
            # Epsilon-greedy action selection
            action, q_value = select_action(trained_model, processed_observation, epsilon, num_actions=num_actions)
            # print(action)

            # Step to next environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            actual_reward.append(reward)  # Accumulate the actual reward
            processed_next_observation = preprocess_observation(next_observation)  # Preprocess the new observation

            processed_observation = processed_next_observation # Push next state to front

        # Post-episode updates
        total_actual_reward = sum(actual_reward)
        actual_episode_rewards.append(total_actual_reward)

        print(f"MsPacman-v0 episode {episode}, reward sum: {total_actual_reward}")

        # Plot and save functionality
        plot_rewards(actual_episode_rewards)

        episode += 1

    env.close()
    plot_rewards(actual_episode_rewards)
    
if __name__=="__main__":
    check_gpu_availability(use_gpus=[0])
    model_number = 875
    model_path = os.path.join(script_dir, 'saved_model', f'mspacman_model_{model_number}_{model_number}')
    MsPacman_RL(model_path)