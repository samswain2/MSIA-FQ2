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

# Preprocess Mrs. Packman Frames
mspacman_color = 210 + 164 + 74
def preprocess_observation(obs):
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.sum(axis=2)  # to greyscale
    img[img==mspacman_color] = 0  # Improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    return img.reshape(88, 80, 1)

# Transform reward
def transform_reward(reward):
    return log(reward, 1000) if reward > 0 else reward
    # return reward

# Build Q-Network
def build_q_network(input_shape=(88, 80, 1), num_actions=9, l2_reg=0.0001):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (8, 8), strides=(2, 2), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (4, 4), strides=(1, 1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        # tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.Dense(num_actions)
    ])
    return model

def adjust_weights(model, target_model, optimizer, states, actions, rewards, next_states, dones, discount_factor, num_actions):
    states, next_states = tf.convert_to_tensor(states, dtype=tf.float32), tf.convert_to_tensor(next_states, dtype=tf.float32)
    rewards, dones, actions = tf.convert_to_tensor(rewards, dtype=tf.float32), tf.convert_to_tensor(dones, dtype=tf.float32), tf.convert_to_tensor(actions, dtype=tf.int32)
    
    with tf.GradientTape() as tape:
        q_values, next_q_values = model(states), target_model(next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * discount_factor * max_next_q_values
        action_mask = tf.one_hot(actions, num_actions, dtype=tf.float32)
        # print(f"Action Mask: {action_mask}")
        predicted_q_values = tf.reduce_sum(q_values * action_mask, axis=1)
        # print(f"Predicted Q Values: {predicted_q_values}")
        loss = tf.reduce_sum(tf.square(target_q_values - predicted_q_values))
        print(f"loss: {loss}")
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Select agent's action
def select_action(model, state, epsilon, num_actions):
    if random.random() < epsilon:
        return random.choice(range(num_actions)), None  # No Q-value in random selection
    q_values = model(tf.expand_dims(state, axis=0))[0]
    return tf.argmax(q_values).numpy(), q_values.numpy()

# Plot model rewards over time
def plot_rewards(episode_rewards, episode_max_q_values, window_size=100):
    # Plot Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label='Episode rewards')
    if len(episode_rewards) >= window_size:
        moving_average = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_average, label=f'{window_size}-episode moving average')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Create artifacts folder if it doesn't exist
    artifacts_path = os.path.join(script_dir, 'artifacts')
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)
    plt.savefig(os.path.join(artifacts_path, 'MsPacman_Progress.png'))
    plt.close()

    # Plot Q-Values
    plt.figure(figsize=(10, 6))
    plt.plot(episode_max_q_values, label='Max Q-values per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Max Q-value')
    plt.legend()
    plt.savefig(os.path.join(artifacts_path, 'MsPacman_Max_Q_Values.png'))
    plt.close()

# Train agent to play MsPacman
def MsPacman_RL():
    # Initialize environment and model
    env = gym.make(
        "MsPacman-v0", 
        # render_mode="human"
        )
    num_actions = 9
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
    model = build_q_network(num_actions=num_actions)
    target_model = build_q_network(num_actions=num_actions)
    target_model.set_weights(model.get_weights())
    transformed_episode_rewards = []
    actual_episode_rewards = []
    episode_q_values = []
    episode = 0

    # Check model
    model.summary()

    # Initialize training control variables
    consecutive_200_rewards = 0
    should_train = True

    # Initialize replay buffer and epsilon for epsilon-greedy action selection
    discount_factor = 0.99
    batch_size = 256
    buffer_size = 50000
    replay_buffer = deque(maxlen=buffer_size)
    epsilon = 1.0
    epsilon_decay = 0.9975
    epsilon_min = 0.01
    episode_max_q_values = []

    # Training loop
    while True:
        observation, info = env.reset()
        processed_observation = preprocess_observation(observation)
        obs_history, reward_history, action_history = [], [], []
        actual_reward = []
        terminated = False
        truncated = False

        accumulated_q_values = []

        # Episode loop
        while not terminated and not truncated:
            # Epsilon-greedy action selection
            action, q_value = select_action(model, processed_observation, epsilon, num_actions=num_actions)
            if q_value is not None:
                accumulated_q_values.append(np.max(q_value))
            obs_history.append(processed_observation)
            action_history.append(action)

            # Step to next environment
            original_action = list(range(num_actions))[action]
            next_observation, reward, terminated, truncated, info = env.step(original_action)
            actual_reward.append(reward)  # Accumulate the actual reward
            reward = transform_reward(reward) # Transform the reward here
            processed_next_observation = preprocess_observation(next_observation)  # Preprocess the new observation
            reward_history.append(reward)

            # Store the experience in the replay buffer
            done_flag = 1 if (terminated or truncated) else 0
            replay_buffer.append((processed_observation, action, reward, processed_next_observation, done_flag))  # Store the preprocessed observations

            if len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)  # Remove the oldest experience if the buffer is full

            processed_observation = processed_next_observation # Push next state to front

        # Record max Q-Values
        max_q_value = max(accumulated_q_values) if accumulated_q_values else 0
        episode_max_q_values.append(max_q_value)

        # Post-episode updates
        total_actual_reward = sum(actual_reward)
        actual_episode_rewards.append(total_actual_reward)
        total_transformed_reward = round(sum(reward_history), 4)
        transformed_episode_rewards.append(total_transformed_reward)

        # Print episode rewards
        moving_num, window = 5000, 100
        if episode >= window-1:
            moving_avg = np.mean(actual_episode_rewards[-window:])
            print(f"MsPacman-v0 episode {episode}, transformed reward sum: {total_transformed_reward}, reward sum: {total_actual_reward}, last {window} avg: {moving_avg:.2f}")
            
            if moving_avg > moving_num:
                print(f"Stopping as the last {window}-episode moving average is greater than {moving_num}")
                saved_model_path = os.path.join(script_dir, 'saved_model')
                if not os.path.exists(saved_model_path):
                    os.makedirs(saved_model_path)
                model.save(os.path.join(saved_model_path, 'mspacman_model'))
                break
        else:
            print(f"MsPacman-v0 episode {episode}, transformed reward sum: {total_transformed_reward}, reward sum: {total_actual_reward}")

        # Plot and save functionality
        if episode % 2 == 0:
            plot_rewards(actual_episode_rewards, episode_max_q_values)
        if episode % 100 == 0:
            model.save(f"saved_model/mspacman_model_{episode}")

        # Adjust model weights
        print(epsilon)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update the Q-network based on the replay buffer
        if len(replay_buffer) >= batch_size and (should_train): # Train model if we have enough data and training flag is set to true
            mini_batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*mini_batch)
            adjust_weights(model, target_model, optimizer, states, actions, rewards, next_states, dones, discount_factor, num_actions=num_actions)

        # Update the target network if the episode number is a multiple of update_target_every
        update_target_every = 10  # Choose an appropriate value
        if episode % update_target_every == 0:
            target_model.set_weights(model.get_weights())

        # Calculate the average Q-value for this episode
        average_q_value = np.mean(accumulated_q_values) if accumulated_q_values else 0
        episode_q_values.append(average_q_value)

        episode += 1

    env.close()
    plot_rewards(actual_episode_rewards, episode_max_q_values)
    
if __name__=="__main__":
    check_gpu_availability()
    MsPacman_RL()