import os
import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import csv

class MsPacmanAgent:

    def __init__(self, config):
        self.config = config

        # Set random seeds for reproducibility
        seed = self.config.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Suppress TensorFlow warnings if required
        if self.config.get('suppress_tf_warnings', True):
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.get_logger().setLevel('ERROR')

        # Check and set GPU availability
        if 'use_gpus' in config:
            self.check_gpu_availability(use_gpus=config['use_gpus'])

        # Initialize the environment
        render_mode = self.config.get('render_mode', None)
        self.env = gym.make("MsPacman-v0", render_mode=render_mode)
        self.num_actions = self.env.action_space.n

        # Build Q-Network models
        self.model = self.build_q_network()
        self.target_model = self.build_q_network()
        self.target_model.set_weights(self.model.get_weights())

        # Training parameters
        self.discount_factor = config.get('discount_factor', 0.99)
        self.batch_size = config.get('batch_size', 256)
        self.replay_buffer = deque(maxlen=config.get('buffer_size', 100000))
        self.entropy_beta = config.get('entropy_beta', 0.01)
        self.update_target_every = config.get('update_target_every', 10)
        self.survival_factor = self.config.get('survival_factor', 0.01)
        self.delta = self.config.get('huber_delta', 1.0)
        self.clip_norm = self.config.get('clip_norm', 1.0)

        # Epsilon parameters for epsilon-greedy policy
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.998)
        self.epsilon_min = config.get('epsilon_min', 0.1)

        # Set the script directory and q_values_filename
        self.script_dir = config.get('script_dir', os.getcwd())
        self.artifact_output_dir = config.get('artifact_output_dir', './artifacts')
        self.model_output_dir = config.get('model_output_dir', './saved_model')
        self.q_values_filename = os.path.join(self.script_dir, self.config.get('q_values_filename', 'mspacman_q_values.txt'))

        # Optimizer
        learning_rate = config.get('learning_rate', 1e-5)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Initialize training statistics
        self.transformed_episode_rewards = []
        self.actual_episode_rewards = []
        self.episode_max_q_values = []
        self.episode_q_values = []
        self.episode_lengths = []
        self.gradient_norms = []
        self.entropy_history = []
        self.action_distribution = np.zeros(self.num_actions)

        # Training control variables
        self.episode = 0
        self.should_train = True

        # Parameters for saving and printing results
        self.average_window = self.config.get('average_window', 100)
        self.stopping_moving_average = self.config.get('stopping_moving_average', 5000)
        self.plot_every = self.config.get('plot_every', 2)
        self.save_model_every = self.config.get('save_model_every', 100)

        # Optionally print the model summary
        if config.get('print_model_summary', False):
            self.model.summary()

    def check_gpu_availability(self, use_gpus=None):
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

    def preprocess_observation(self, obs, mspacman_color = (210 + 164 + 74)):
        img = obs[1:176:2, ::2]  # crop and downsize
        img = img.sum(axis=2)  # to greyscale
        img[img==mspacman_color] = 0  # Improve contrast
        img = (img / 255.0).astype(np.float32)  # Normalize between 0 and 1
        return img.reshape(88, 80, 1)

    def transform_reward(self, reward):
        """
        Transforms the reward for further processing.
        """
        survival_reward = self.survival_factor * self.step_counter
        return reward + survival_reward

    def build_q_network(self, input_shape=(88, 80, 1), num_actions=9, l2_reg=0.0001):
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

    def adjust_weights(self, model, target_model, optimizer, states, actions, rewards, next_states, dones, discount_factor, num_actions, entropy_beta, delta, clip_norm):
        states, next_states = tf.convert_to_tensor(states, dtype=tf.float32), tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards, dones, actions = tf.convert_to_tensor(rewards, dtype=tf.float32), tf.convert_to_tensor(dones, dtype=tf.float32), tf.convert_to_tensor(actions, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            q_values, next_q_values = model(states), target_model(next_states)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)
            target_q_values = rewards + (1 - dones) * discount_factor * max_next_q_values
            action_mask = tf.one_hot(actions, num_actions, dtype=tf.float32)
            predicted_q_values = tf.reduce_sum(q_values * action_mask, axis=1)

            # Entropy bonus implementation
            action_probs = tf.nn.softmax(q_values)
            action_log_probs = tf.math.log(action_probs + 1e-5)
            entropy = -tf.reduce_sum(action_probs * action_log_probs, axis=-1)
            entropy_bonus = entropy_beta * entropy
            self.entropy_history.append(tf.reduce_mean(entropy).numpy())

            # Huber loss calculation
            huber_loss = tf.keras.losses.Huber(delta=delta)
            loss_per_example = huber_loss(target_q_values, predicted_q_values)
            adjusted_loss_per_example = loss_per_example - entropy_bonus
            loss = tf.reduce_mean(adjusted_loss_per_example)
            print(f"loss: {loss}")
            
        gradients = tape.gradient(loss, model.trainable_variables)
        gradient_norms = [tf.norm(g).numpy() for g in gradients]  # Calculate norms of gradients
        average_gradient_norm = np.mean(gradient_norms)  # Compute average gradient norm
        self.gradient_norms.append(average_gradient_norm)  # Log the average gradient norm
        gradients = [tf.clip_by_norm(g, clip_norm) for g in gradients] # Apply gradient clipping
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def select_action(self, model, state, epsilon, num_actions):
        if random.random() < epsilon:
            return random.choice(range(num_actions)), None  # No Q-value in random selection
        
        q_values = model(tf.expand_dims(state, axis=0))[0]
        return tf.argmax(q_values).numpy(), q_values.numpy()

    def plot_and_save(self, data, y_label, file_name, window_size, artifacts_path):
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

    def plot_rewards(self, episode_rewards, episode_max_q_values, window_size=100):
        # Ensure the artifacts directory exists
        artifacts_path = os.path.join(self.script_dir, 'artifacts')
        os.makedirs(artifacts_path, exist_ok=True)
        
        # Plot and save rewards and Q-values
        self.plot_and_save(episode_rewards, 'Total Reward', 'MsPacman_Progress.png', window_size, artifacts_path)
        self.plot_and_save(episode_max_q_values, 'Max Q-value', 'MsPacman_Max_Q_Values.png', window_size, artifacts_path)

    def print_and_save_episode_results(self):
        window = self.config.get('average_window', 100)
        moving_num = self.config.get('stopping_moving_average', 5000)

        if self.episode >= window - 1:
            moving_avg = np.mean(self.actual_episode_rewards[-window:])
            print(f"MsPacman-v0 episode {self.episode}, transformed reward sum: {self.transformed_episode_rewards[-1]}, reward sum: {self.actual_episode_rewards[-1]}, last {window} avg: {moving_avg:.2f}")
            
            if moving_avg > moving_num:
                print(f"Stopping as the last {window}-episode moving average is greater than {moving_num}")
                self.save_model('mspacman_model', final=True)
                self.should_train = False  # To stop script
        else:
            print(f"MsPacman-v0 episode {self.episode}, transformed reward sum: {self.transformed_episode_rewards[-1]}, reward sum: {self.actual_episode_rewards[-1]}")

        # Plot and save functionality
        if self.episode % self.config.get('plot_every', 2) == 0:
            self.plot_rewards(self.actual_episode_rewards, self.episode_max_q_values)
        if self.episode % self.config.get('save_model_every', 100) == 0:
            self.save_model(f"mspacman_model_{self.episode}")

    def save_model(self, model_name, final=False):
        saved_model_path = os.path.join(self.script_dir, 'saved_model')
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)
        if final:
            self.model.save(os.path.join(saved_model_path, model_name))
        else:
            self.model.save(os.path.join(saved_model_path, f"{model_name}_{self.episode}"))

    def save_metrics(self, filename, episode, episode_metrics, all_q_values):
        # Convert all Q-values to a string, separating Q-values by commas and actions by semicolons
        all_q_values_str = ';'.join([','.join(map(str, q)) if not isinstance(q, str) else q for q in all_q_values])

        # Check if it's the first episode and the file exists, then remove it
        if episode == 0 and os.path.isfile(filename):
            os.remove(filename)
        
        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as csvfile:
            headers = [
                'episode',           # Identifier of the episode
                'epsilon',           # Parameter that might define the behavior policy in a learning algorithm
                'actions',           # Outcome: What actions were taken
                'episode_length',    # Outcome: Duration or length of the episode
                'sum_reward',        # Outcome: Total reward accumulated in the episode
                'average_q_value',   # Performance metric: Average of the Q values
                'average_gradient_norm', # Performance metric: Average norm of the gradients
                'average_entropy',   # Performance metric: Average entropy, indicating randomness of action selection
                'all_q_values'       # Detailed log: All Q values for all actions taken, for deeper analysis
            ]
            writer = csv.DictWriter(csvfile, fieldnames=headers, quoting=csv.QUOTE_MINIMAL)

            if not file_exists:
                writer.writeheader()  # Write the header only once

            # Include the more variables in the metrics to be written
            episode_metrics['episode'] = episode
            episode_metrics['all_q_values'] = '"' + all_q_values_str.replace('"', '""') + '"'
            writer.writerow(episode_metrics)

    def train_agent(self):
        """
        Main function to train the MsPacman agent using reinforcement learning.
        """
        # Training loop
        while self.should_train:
            observation, info = self.env.reset()
            processed_observation = self.preprocess_observation(obs=observation)
            obs_history, reward_history, action_history = [], [], []
            actual_reward = []
            all_q_values = []
            accumulated_q_values = []
            self.step_counter = 0
            
            terminated = False
            truncated = False

            # Episode loop
            while not terminated and not truncated:
                # Epsilon-greedy action selection
                action, q_value = self.select_action(model=self.model, state=processed_observation, epsilon=self.epsilon, num_actions=self.num_actions)
                if q_value is not None:
                    accumulated_q_values.append(np.max(q_value))
                    all_q_values.append(q_value)
                else:
                    all_q_values.append("Random action")
                obs_history.append(processed_observation)
                action_history.append(action)
                self.action_distribution[action] += 1

                # Step to next environment
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                self.step_counter += 1 # Increment the step counter
                actual_reward.append(reward)  # Accumulate the actual reward
                reward = self.transform_reward(reward) # Transform the reward here
                processed_next_observation = self.preprocess_observation(obs=next_observation)  # Preprocess the new observation
                reward_history.append(reward)

                # Store the experience in the replay buffer
                done_flag = 1 if (terminated or truncated) else 0
                self.replay_buffer.append((processed_observation, action, reward, processed_next_observation, done_flag))  # Store the preprocessed observations

                processed_observation = processed_next_observation # Push next state to front

            # Record max Q-Values
            max_q_value = np.mean(accumulated_q_values) if accumulated_q_values else 0
            self.episode_max_q_values.append(max_q_value)

            # Post-episode updates
            total_actual_reward = sum(actual_reward)
            self.actual_episode_rewards.append(total_actual_reward)
            total_transformed_reward = round(sum(reward_history), 4)
            self.transformed_episode_rewards.append(total_transformed_reward)

            # Print episode rewards
            self.print_and_save_episode_results()

            # Adjust model weights
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"Epsilon: {self.epsilon}")
            if len(self.replay_buffer) >= self.batch_size and self.should_train:
                # Sample a minibatch from the replay buffer
                mini_batch = random.sample(self.replay_buffer, self.batch_size)
                states, actions, rewards, next_states, dones = zip(*mini_batch)
                self.adjust_weights(
                    self.model, self.target_model, self.optimizer, 
                    states, actions, rewards, next_states, dones, 
                    self.discount_factor, self.num_actions, self.entropy_beta,
                    delta=self.delta, clip_norm=self.clip_norm
                    )

            # Update the target network
            if self.episode % self.update_target_every == 0:
                self.target_model.set_weights(self.model.get_weights())

            # Calculate the average Q-value for this episode
            average_q_value = np.mean(accumulated_q_values) if accumulated_q_values else 0
            self.episode_q_values.append(average_q_value)

            episode_metrics = {
                'sum_reward': np.sum(actual_reward),
                'average_q_value': np.mean(accumulated_q_values) if accumulated_q_values else 0,
                'average_gradient_norm': self.gradient_norms[-1],
                'average_entropy': self.entropy_history[-1],
                'epsilon': self.epsilon,
                'episode_length': self.step_counter,
                'actions': action_history
            }

            # Call the save_metrics method and pass all_q_values list
            self.save_metrics('training_metrics.csv', self.episode, episode_metrics, all_q_values)

            self.episode_lengths.append(self.step_counter)
            self.episode += 1

        self.env.close()
        self.plot_rewards(self.actual_episode_rewards, self.episode_max_q_values)

if __name__ == "__main__":
    config = {
        'seed': 42,  # Used to set the random seed for reproducibility
        'suppress_tf_warnings': True,  # Determines if TensorFlow warnings should be suppressed
        'render_mode': None,  # The mode used by the gym environment for rendering
        'script_dir': os.getcwd(),  # Directory for saving artifacts and models
        'artifact_output_dir': './artifacts',  # Directory for saving plots and figures
        'model_output_dir': './saved_model',  # Directory where the model will be saved
        'q_values_filename': 'mspacman_q_values.txt',  # File name for saving Q-values
        'use_gpus': [1], # List of GPU indices to use
        'num_actions': 9,  # Number of possible actions in the environment (should match the environment's action space)
        'learning_rate': 1e-5,  # Learning rate for the optimizer
        'buffer_size': 100000,  # Maximum size of the replay buffer
        'batch_size': 512,  # Size of the batch when sampling from the replay buffer
        'discount_factor': 0.99,  # Discount factor for future rewards
        'epsilon_start': 1.0,  # Starting value for epsilon in the epsilon-greedy policy
        'epsilon_decay': 0.999,  # Decay rate for epsilon after each episode
        'epsilon_min': 0.025,  # Minimum value for epsilon
        'entropy_beta': 0.01,  # Scaling factor for the entropy bonus
        # 'survival_factor': 0.001, # Factor to calculate the reward based on survival time
        'survival_factor': 0.000, # Factor to calculate the reward based on survival time
        'huber_delta': 1.0,  # The delta value for the Huber loss, adjust as needed
        'clip_norm': 1.0,  # The clip norm value for gradient clipping, adjust as needed
        'update_target_every': 25,  # Number of episodes between updating the target network
        'average_window': 100,  # Window size for calculating the moving average of rewards
        'stopping_moving_average': 5000,  # Moving average threshold for stopping the training
        'plot_every': 2,  # Frequency (in episodes) for plotting the training progress
        'save_model_every': 25,  # Frequency (in episodes) for saving the model checkpoints
        'print_model_summary': True,  # Whether to print the model summary
    }
    agent = MsPacmanAgent(config)
    agent.train_agent()