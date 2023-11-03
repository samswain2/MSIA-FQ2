import gym
import warnings
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
import os
import tensorflow as tf
from collections import deque
import math

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def check_gpu_availability(use_gpus=None):
    """
    Prints available GPUs and restricts TensorFlow to use only specified GPUs by indices.
    
    Args:
    use_gpus (list of int, optional): GPU indices to use. Defaults to None, using all available GPUs.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return print("GPU is not available, using CPU instead.")
    
    print(f"GPUs available: {gpus}")
    if use_gpus is None: return
    
    try:
        tf.config.set_visible_devices([gpus[i] for i in use_gpus], 'GPU')
        print(f"{len(use_gpus)} GPUs set to be used.")
    except RuntimeError as e:
        print(f"Error setting visible devices: {e}")

def preprocess_observation(obs, mspacman_color = (210 + 164 + 74)):
    """
    Preprocess the Ms. Pacman game frames by cropping, downsizing, converting to greyscale,
    improving contrast by setting a specific color to zero, normalizing pixel values, and 
    reshaping for the neural network input.

    Args:
    obs (numpy array): The original RGB game frame.

    Returns:
    numpy array: The preprocessed game frame as a greyscale image.
    """
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.sum(axis=2)  # to greyscale
    img[img==mspacman_color] = 0  # Improve contrast
    img = (img / 255.0).astype(np.float32)  # Normalize between 0 and 1
    return img.reshape(88, 80, 1)

def transform_reward(reward):
    """
    Transforms the reward for further processing. Currently a placeholder function.

    Args:
        reward (float): The original reward from the environment.

    Returns:
        float: The transformed reward. Currently returns the unmodified reward.
    """
    return reward

def build_q_network(input_shape=(88, 80, 1), num_actions=9, l2_reg=0.0001):
    """
    Constructs a dueling Deep Q-Network model with convolutional layers.

    This network takes an input state of the game and outputs Q-values for each action.
    It uses a dueling architecture, where the network estimates the value of being in a
    given state, and the advantage of each action separately. The final Q-values are
    computed by combining these two streams.

    Args:
    input_shape (tuple, optional): The shape of the input state. Defaults to (88, 80, 1).
    num_actions (int, optional): The number of possible actions. Defaults to 9.
    l2_reg (float, optional): L2 regularization factor. Defaults to 0.0001.

    Returns:
    tf.keras.Model: The constructed Q-network model.
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (8, 8), strides=(2, 2), activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=(1, 1), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)

    # Dueling DQN branches
    value_stream = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    advantage_stream = tf.keras.layers.Dense(num_actions, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    q_values = tf.keras.layers.Add()([value_stream, advantage_stream - tf.reduce_mean(advantage_stream, axis=1, keepdims=True)])

    model = tf.keras.Model(inputs=inputs, outputs=q_values)

    return model

def adjust_weights(model, target_model, optimizer, states, actions, rewards, next_states, dones, discount_factor, num_actions, entropy_beta):
    """
    Updates the weights of the given model based on the training batch.
    
    This function calculates the target Q-values, computes the loss with an entropy bonus for exploration,
    and performs a gradient update on the model weights.
    
    Args:
        model (tf.keras.Model): The main Q-network model.
        target_model (tf.keras.Model): The target Q-network model.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer for training.
        states (list): The list of state tensors.
        actions (list): The list of action tensors.
        rewards (list): The list of reward tensors.
        next_states (list): The list of next state tensors.
        dones (list): The list of done flags (1 for terminal state, else 0).
        discount_factor (float): The discount factor for future rewards.
        num_actions (int): The number of possible actions.
        entropy_beta (float): The scaling factor for the entropy bonus.
    """
    states, next_states = tf.convert_to_tensor(states, dtype=tf.float32), tf.convert_to_tensor(next_states, dtype=tf.float32)
    rewards, dones, actions = tf.convert_to_tensor(rewards, dtype=tf.float32), tf.convert_to_tensor(dones, dtype=tf.float32), tf.convert_to_tensor(actions, dtype=tf.int32)
    
    with tf.GradientTape() as tape:
        q_values, next_q_values = model(states), target_model(next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * discount_factor * max_next_q_values
        action_mask = tf.one_hot(actions, num_actions, dtype=tf.float32)
        predicted_q_values = tf.reduce_sum(q_values * action_mask, axis=1)

        # Entropy implementation
        action_probs = tf.nn.softmax(q_values)
        action_log_probs = tf.math.log(action_probs + 1e-5)
        entropy = -tf.reduce_sum(action_probs * action_log_probs, axis=-1)
        entropy_bonus = entropy_beta * entropy
        loss = tf.reduce_mean(tf.square(target_q_values - predicted_q_values) - entropy_bonus)
        print(f"loss: {loss}")
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def select_action(model, state, epsilon, num_actions):
    """
    Selects an action using an epsilon-greedy policy.
    
    With a probability of `epsilon`, a random action is chosen (exploration).
    Otherwise, the action with the highest Q-value predicted by the model is chosen (exploitation).
    
    Args:
        model (tf.keras.Model): The neural network model that predicts Q-values.
        state (np.array): The current state representation.
        epsilon (float): The probability of choosing a random action.
        num_actions (int): The total number of possible actions.
    
    Returns:
        int: The action selected.
        np.array or None: The Q-values predicted by the model or None if a random action is chosen.
    """
    if random.random() < epsilon:
        return random.choice(range(num_actions)), None  # No Q-value in random selection
    
    q_values = model(tf.expand_dims(state, axis=0))[0]
    return tf.argmax(q_values).numpy(), q_values.numpy()

def plot_and_save(data, y_label, file_name, window_size, artifacts_path):
    """
    Plots and saves a given data set.

    Args:
        data (list): The data points to plot.
        y_label (str): The label for the Y-axis.
        file_name (str): The file name to save the plot as.
        window_size (int): The size of the window for the moving average.
        artifacts_path (str): Path to the directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data, label=f'{y_label} per Episode')
    if len(data) >= window_size:
        moving_average = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(data)), moving_average, label=f'{window_size}-episode moving average')
    plt.xlabel('Episodes')
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join(artifacts_path, file_name))
    plt.close()

def plot_rewards(episode_rewards, episode_max_q_values, window_size=100):
    """
    Plots and saves the model's rewards and maximum Q-values over time.

    Args:
        episode_rewards (list): List of total rewards per episode.
        episode_max_q_values (list): List of maximum Q-values per episode.
        window_size (int): Size of the moving window for average calculation.
    """
    # Ensure the artifacts directory exists
    artifacts_path = os.path.join(script_dir, 'artifacts')
    os.makedirs(artifacts_path, exist_ok=True)
    
    # Plot and save rewards and Q-values
    plot_and_save(episode_rewards, 'Total Reward', 'MsPacman_Progress.png', window_size, artifacts_path)
    plot_and_save(episode_max_q_values, 'Max Q-value', 'MsPacman_Max_Q_Values.png', window_size, artifacts_path)

def MsPacman_RL():
    """
    Main function to train the MsPacman agent using reinforcement learning.
    """
    # Save training info
    q_values_filename = os.path.join(script_dir, 'mspacman_q_values.txt')

    # Initialize environment and model
    env = gym.make(
        "MsPacman-v0", 
        # render_mode="human"
        )
    num_actions = 9
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    model = build_q_network(num_actions=num_actions)
    target_model = build_q_network(num_actions=num_actions)
    target_model.set_weights(model.get_weights())
    transformed_episode_rewards = []
    actual_episode_rewards = []
    episode_q_values = []
    # state_visit_counts = {} # Initialize a dictionary to keep track of state visit counts
    episode = 0

    # Check model
    model.summary()

    # Initialize training control variables
    consecutive_200_rewards = 0
    should_train = True

    # Initialize replay buffer and epsilon for epsilon-greedy action selection
    discount_factor = 0.99
    batch_size = 256
    buffer_size = 100000
    replay_buffer = deque(maxlen=buffer_size)
    epsilon = 1.0
    epsilon_decay = 0.998
    epsilon_min = 0.1
    episode_max_q_values = []
    entropy_beta = 0.01

    # Training loop
    while True:
        observation, info = env.reset()
        processed_observation = preprocess_observation(observation)
        obs_history, reward_history, action_history = [], [], []
        actual_reward = []
        all_q_values = []
        terminated = False
        truncated = False

        accumulated_q_values = []

        # Episode loop
        while not terminated and not truncated:
            # Epsilon-greedy action selection
            action, q_value = select_action(model, processed_observation, epsilon, num_actions=num_actions)
            if q_value is not None:
                accumulated_q_values.append(np.max(q_value))
                all_q_values.append(q_value)
            else:
                all_q_values.append("Random action")
            obs_history.append(processed_observation)
            action_history.append(action)

            # Step to next environment
            original_action = list(range(num_actions))[action]
            next_observation, reward, terminated, truncated, info = env.step(original_action)
            actual_reward.append(reward)  # Accumulate the actual reward
            # reward += exploration_bonus(processed_observation) # Apply the exploration bonus to the reward
            reward = transform_reward(reward) # Transform the reward here
            processed_next_observation = preprocess_observation(next_observation)  # Preprocess the new observation
            reward_history.append(reward)

            # Store the experience in the replay buffer
            done_flag = 1 if (terminated or truncated) else 0
            replay_buffer.append((processed_observation, action, reward, processed_next_observation, done_flag))  # Store the preprocessed observations

            processed_observation = processed_next_observation # Push next state to front

        # Record max Q-Values
        max_q_value = np.mean(accumulated_q_values) if accumulated_q_values else 0
        episode_max_q_values.append(max_q_value)

        # Save all Q-values to a text file every 10 episodes
        if episode % 10 == 0:
            with open(q_values_filename, 'a') as file:
                file.write(f'Episode {episode}:\n')
                for iteration, q_values in enumerate(all_q_values):
                    action = action_history[iteration]
                    q_values_str = ' '.join(map(str, q_values)) if not isinstance(q_values, str) else q_values
                    file.write(f'Iteration {iteration}: Action: {action}, Q-values: {q_values_str}\n')
                file.write('\n')

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
        print(f"Epsilon: {epsilon}")
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update the Q-network based on the replay buffer
        if len(replay_buffer) >= batch_size and (should_train): # Train model if we have enough data and training flag is set to true
            mini_batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*mini_batch)
            adjust_weights(model, target_model, optimizer, states, actions, rewards, next_states, dones, discount_factor, num_actions, entropy_beta)

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
    check_gpu_availability(use_gpus=[1])
    MsPacman_RL()