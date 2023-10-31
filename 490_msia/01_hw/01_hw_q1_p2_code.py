import gym
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def check_gpu_availability():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available")
        for gpu in gpus:
            print(f"GPU device: {gpu}")
    else:
        print("GPU is not available")

def build_model(input_shape=(4,), num_choices=2, reg=0.00001):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg))(input_layer)
    # x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg))(x)
    action_probs = tf.keras.layers.Dense(num_choices, activation="softmax", name='action')(x)
    value_function = tf.keras.layers.Dense(1, name='value')(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[action_probs, value_function])
    return model

def select_action(model, observation, num_choices=2):
    action_probs, _ = model(np.array([observation]))
    action_probs = action_probs.numpy()
    # print(action_probs)
    return np.random.choice(num_choices, p=action_probs[0])

def compute_discounted_rewards(reward_history, discount_factor=0.95):
    discounted_rewards, cumulative_reward = [], 0
    for reward in reversed(reward_history):
        cumulative_reward = reward + discount_factor * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    return discounted_rewards

def adjust_weights(model, optimizer, obs_history, action_history, discounted_rewards):
    with tf.GradientTape() as tape:
        action_probs, values = model(np.array(obs_history))
        values = tf.squeeze(values)
        
        baseline = values.numpy()
        adjusted_discounted_rewards = np.array(discounted_rewards) - baseline
        
        indices = tf.stack([tf.range(len(action_history), dtype=tf.int32), tf.convert_to_tensor(action_history, dtype=tf.int32)], axis=1)
        chosen_probs = tf.gather_nd(action_probs, indices)
        
        loss_policy = -tf.math.log(chosen_probs) * adjusted_discounted_rewards
        loss_value = tf.sqrt(tf.square(values - discounted_rewards))
        
        loss = tf.reduce_mean(loss_policy + loss_value)
        
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def plot_rewards(episode_rewards, window_size=100):
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label='Episode rewards')
    if len(episode_rewards) >= window_size:
        moving_average = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_average, label=f'{window_size}-episode moving average')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig('artifacts/CartPole_Progress.png')
    plt.close()

def CartPole_RL():
    env = gym.make("CartPole-v0", 
                #    render_mode="human"
                   )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model = build_model()
    episode_rewards = []
    episode = 0

    consecutive_200_rewards = 0  # Count number of 200 occurences
    should_train = True  # Initialize flag for training

    while True:
        observation, info = env.reset()
        obs_history, reward_history, action_history = [], [], []

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = select_action(model, observation)
            obs_history.append(observation)
            action_history.append(action)

            observation, reward, terminated, truncated, info = env.step(action)
            reward_history.append(reward)

        discounted_rewards = compute_discounted_rewards(reward_history)

        total_reward = sum(reward_history)
        episode_rewards.append(total_reward)

        moving_num, window = 195, 100
        if episode >= window-1:
            moving_avg = np.mean(episode_rewards[-window:])
            print(f"CartPole-v0 episode {episode}, reward sum: {total_reward}, last {window} avg: {moving_avg:.2f}")
            
            if moving_avg > moving_num:
                print(f"Stopping as the last {window}-episode moving average is greater than {moving_num}")
                break
        else:
            print(f"CartPole-v0 episode {episode}, reward sum: {total_reward}")


        ### TESTING PLOT AND SAVE FUNCTIONALITY
        # Add code here to plot every 10 episodes
        if episode % 25 == 0:
            plot_rewards(episode_rewards)

        # Add code here to save the model every 100 episodes
        if episode % 100 == 0:
            model.save(f"saved_model/cartpole_model_{episode}")
        ### TESTING PLOT AND SAVE FUNCTIONALITY


        ### TRAINING STOP

        if total_reward == 200:  # Check for consecutive rewards of 200
            consecutive_200_rewards += 1
            if consecutive_200_rewards >= 50 and should_train == True:
                print("Stopping training as the reward has been 200 for 50 episodes in a row")
                should_train = False  # Set flag to False

                # Check if folder exists, if not create it
                if not os.path.exists("saved_model"):
                    os.makedirs("saved_model")

                # Save the model
                model.save("saved_model/cartpole_model")
                
        else:
            consecutive_200_rewards = 0  # Reset the counter if the reward is not 200

        # Modify the weight adjustment to respect the should_train flag
        if should_train:
            adjust_weights(model, optimizer, obs_history, action_history, discounted_rewards)

        episode += 1

    env.close()

    plot_rewards(episode_rewards)

if __name__ == "__main__":
    check_gpu_availability()
    CartPole_RL()
