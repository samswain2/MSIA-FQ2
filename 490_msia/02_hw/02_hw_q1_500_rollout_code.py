import gym
import warnings
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
import os
import tensorflow as tf

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def check_gpu_availability():
    # List all GPUs available to TensorFlow
    gpus = tf.config.list_physical_devices('GPU')

    # Check if any GPUs are available
    if len(gpus) > 0:
        print("GPU is available")
        for gpu in gpus:
            print(f"GPU device: {gpu}")
    else:
        print("GPU is not available")

# Select agent's action
def select_action(model, state, epsilon=0.01):
    if random.random() < epsilon:
        return random.choice([0, 1])
    q_values = model(tf.expand_dims(state, axis=0))[0]
    return tf.argmax(q_values).numpy()

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
    stats_file_path = os.path.join(artifacts_path, 'CartPole_500_Rollout_Stats.txt')
    with open(stats_file_path, 'w') as f:
        f.write(f"Mean Reward: {mean_reward}\n")
        f.write(f"Standard Deviation: {std_reward}\n")
    
    plt.savefig(os.path.join(artifacts_path, 'CartPole_500_Rollout_Progress.png'))
    plt.close()

# Evaluate trained agent
def evaluate_trained_model(model_path, rollout_len=500):
    # Initialize environment and model
    env = gym.make("CartPole-v0", 
                #    render_mode="human"
                   )
    trained_model = tf.keras.models.load_model(model_path)
    episode_rewards = []

    # Training loop
    for episode in range(1, rollout_len+1):
        observation, info = env.reset()
        obs_history, reward_history, action_history = [], [], []
        terminated = False
        truncated = False

        # Episode loop
        while not terminated and not truncated:
            # Epsilon-greedy action selection
            action = select_action(trained_model, observation)

            # Step to next environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            reward_history.append(reward)

            observation = next_observation

        # Post-episode updates
        total_reward = sum(reward_history)
        episode_rewards.append(total_reward)

        print(f"CartPole-v0 episode {episode}, reward sum: {total_reward}")

        # Plot and save functionality
        if episode % 10 == 0:
            plot_rewards(episode_rewards)

    env.close()
    plot_rewards(episode_rewards)

if __name__ == "__main__":
    check_gpu_availability()
    model_path = os.path.join(script_dir, 'saved_model', 'cartpole_model')
    evaluate_trained_model(model_path, rollout_len=500)