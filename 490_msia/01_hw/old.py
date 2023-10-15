import gym
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# List all GPUs available to TensorFlow
gpus = tf.config.list_physical_devices('GPU')

# Check if any GPUs are available
if len(gpus) > 0:
    print("GPU is available")
    for gpu in gpus:
        print(f"GPU device: {gpu}")
else:
    print("GPU is not available")

# Initialize Environment and Model
env = gym.make("CartPole-v0")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
])

def CartPole_RL(model, discount_factor=0.95, num_choices=2):
    episode_rewards = []
    episode = 0
    
    while True:
        observation, info = env.reset()
        obs_history, reward_history, action_history = [], [], []

        terminated = False
        truncated = False

        while not terminated and not truncated:
            probabilities = model.predict(np.array([observation]), verbose=0)[0]
            action = np.random.choice(num_choices, p=probabilities)

            obs_history.append(observation)
            action_history.append(action)

            observation, reward, terminated, truncated, info = env.step(action)
            reward_history.append(reward)

        # Discount rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(reward_history):
            cumulative_reward = reward + discount_factor * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        def adjust_weights(obs_history, action_history, discounted_rewards):
            with tf.GradientTape() as tape:
                action_probabilities = model(np.array(obs_history))
                indices = list(zip(range(len(action_history)), action_history))
                chosen_action_probs = tf.gather_nd(action_probabilities, indices)
                loss = -tf.math.log(chosen_action_probs) * discounted_rewards
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        adjust_weights(obs_history, action_history, discounted_rewards)

        total_reward = sum(reward_history)
        episode_rewards.append(total_reward)

        moving_num, window = 195, 100
        if episode >= window-1:
            moving_avg = np.mean(episode_rewards[-window:]) # Compute moving average
            print(f"CartPole-v0 episode {episode}, reward sum: {total_reward}, last {window} avg: {moving_avg:.2f}")
            
            if moving_avg > moving_num:
                print(f"Stopping as the last {window}-episode moving average is greater than {moving_num}")
                break
        
        else:
            print(f"CartPole-v0 episode {episode}, reward sum: {total_reward}")

        episode += 1

    env.close()

    # Save Results
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig('artifacts/episode_rewards.png')

if __name__ == "__main__":
    CartPole_RL(model)
