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
        print("GPU is available", flush=True)
        for gpu in gpus:
            print(f"GPU device: {gpu}", flush=True)
            tf.config.experimental.set_memory_growth(gpu, True)

        # os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set the second GPU as available #
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Filter out warnings

    else:
        print("GPU is not available", flush=True)

def preprocess(image, downsample_factor=2):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::downsample_factor,::downsample_factor,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1
    # In preprocess function
    return tf.reshape(tf.cast(image, tf.float32), [80, 80, 1])


# def build_model(input_shape=(80, 80, 1), num_choices=2, reg=0.0001):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Flatten(input_shape=input_shape),  # Set input_shape here
#         tf.keras.layers.Dense(256, activation='relu'),  # Dense hidden layer
#         tf.keras.layers.Dense(num_choices, activation="softmax")  # Dense output layer
#     ])
#     return model 

# def build_model(input_shape=(80, 80, 1), num_choices=2, reg=0.0001):
#     input_layer = tf.keras.layers.Input(shape=input_shape)
    
#     x = tf.keras.layers.Flatten()(input_layer)
#     x = tf.keras.layers.Dense(256, activation='relu')(x)
#     x = tf.keras.layers.Dense(128, activation='relu')(x)

#     action_probs = tf.keras.layers.Dense(num_choices, activation="softmax", name='action')(x)
#     state_value = tf.keras.layers.Dense(1, activation=None, name='value')(x)
    
#     model = tf.keras.Model(inputs=input_layer, outputs=[action_probs, state_value])

#     return model

def build_model(input_shape=(80, 80, 1), num_choices=2, reg=0.0001):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    
    # Convolutional layers
    x = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_layer)
    # x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    # x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    # Fully connected layers
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg))(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg))(x)
    
    # Output layers
    action_probs = tf.keras.layers.Dense(num_choices, activation="softmax", name='action')(x)
    state_value = tf.keras.layers.Dense(1, activation=None, name='value')(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[action_probs, state_value])

    return model




# def select_action(model, observation):
#     probabilities = model(tf.expand_dims(observation, axis=0))[0].numpy()
#     action_idx = np.random.choice([0, 1], p=probabilities)
#     return action_idx


def select_action(model, observation):
    action_probs, _ = model(tf.expand_dims(observation, axis=0))
    action_probs = action_probs[0].numpy()
    action_idx = np.random.choice([0, 1], p=action_probs)
    return action_idx



def compute_discounted_rewards(reward_history, discount_factor=0.99):
    discounted_rewards, cumulative_reward = [], 0
    for reward in reversed(reward_history):
        cumulative_reward = reward + discount_factor * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    
    # Normalization of discounted rewards
    mean = np.mean(discounted_rewards)
    std = np.std(discounted_rewards)
    normalized_rewards = (discounted_rewards - mean) / (std + 1e-8)  # Added epsilon to avoid division by zero
    
    return normalized_rewards

# def adjust_weights(model, optimizer, obs_history, action_history, discounted_rewards):
#     # Added print statement to observe the incoming discounted_rewards
#     # print(f"Initial discounted_rewards: {discounted_rewards}")

#     # print("Getting discounted rewards", flush=True)
#     discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

#     # Added print statement to observe the tensor-converted discounted_rewards
#     # print(f"Tensor-converted discounted_rewards: {discounted_rewards}")

#     with tf.GradientTape() as tape:
#         # print("Getting probs", flush=True)
#         probs = model(tf.convert_to_tensor(obs_history, dtype=tf.float32))

#         # Added print statement to observe the model probabilities
#         # print(f"Model probabilities: {probs}")

#         # print("Getting Indices", flush=True)
#         indices = tf.stack([tf.range(len(action_history), dtype=tf.int32), tf.convert_to_tensor(action_history, dtype=tf.int32)], axis=1)

#         # Added print statement to observe the indices
#         # print(f"indices: {indices}")

#         # print("Getting Chosen Probs", flush=True)
#         chosen_probs = tf.gather_nd(probs, indices)

#         # Added print statement to observe the chosen probabilities
#         # print(f"Chosen probabilities: {chosen_probs}")
#         # print(f"Chosen probabilities: {chosen_probs[:4].numpy()}")

#         # print("Getting Loss", flush=True)
#         loss = -tf.math.log(chosen_probs) * discounted_rewards
#         loss = tf.reduce_sum(loss)

#         # Added print statement to observe the loss
#         print(f"Loss: {loss}")

#     # print("Applying Gradient and Optimizer", flush=True)
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))

#     # Added print statement to observe the applied gradients
#     # print(f"Applied gradients: {grads}")

def adjust_weights(model, optimizer, obs_history, action_history, discounted_rewards):
    discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
    with tf.GradientTape() as tape:
        action_probs, values = model(tf.convert_to_tensor(obs_history, dtype=tf.float32))
        values = tf.squeeze(values)
        
        indices = tf.stack([tf.range(len(action_history), dtype=tf.int32), tf.convert_to_tensor(action_history, dtype=tf.int32)], axis=1)
        chosen_probs = tf.gather_nd(action_probs, indices)
        
        advantage = discounted_rewards - values
        
        policy_loss = -tf.math.log(chosen_probs) * advantage
        value_loss = 0.5 * tf.math.square(advantage)
        
        loss = tf.reduce_sum(policy_loss + value_loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))




def save_plot(episode_rewards, episode):
    window = 100  # Size of the window for calculating moving average
    moving_avgs = [np.mean(episode_rewards[max(0, i - window + 1):i + 1]) for i in range(len(episode_rewards))]
    
    plt.plot(episode_rewards, label='Total Reward')
    plt.plot(moving_avgs, label=f'{window}-Episode Moving Average')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig(f'artifacts/Pong_Progress.png')
    plt.close()

def Pong_RL():
    env = gym.make("Pong-v0", 
                #    render_mode="human"
                   )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    iter_number = 2000

    ### Code below will load existing model in

    model_path = f"saved_model/pong_model_{iter_number}"  # Adjust this to your specific model path

    # Attempt to load the model
    if os.path.exists(model_path):
        print("Loading existing model from", model_path, flush=True)
        model = tf.keras.models.load_model(model_path)
    else:
        print("Building a new model", flush=True)
        model = build_model()

    # model = build_model()
    model.summary()
    episode_rewards = []
    episode = iter_number

    consecutive_21_rewards = 0  # Count number of 21 occurences
    should_train = True  # Initialize flag for training

    while True:
        observation, info = env.reset()
        obs_history, reward_history, action_history = [], [], []

        terminated = False
        truncated = False

        # print("Starting Game", flush=True)

        while not terminated and not truncated:
            processed_observation = preprocess(observation)
            action = select_action(model, processed_observation)
            obs_history.append(processed_observation)
            action_history.append(action)

            action_to_take = 2 if action == 0 else 3  # Remap the action index to the appropriate Pong-v0 action
            observation, reward, terminated, truncated, info = env.step(action_to_take)
            
            reward_history.append(reward)

        # print("Finishing Game", flush=True)

        discounted_rewards = compute_discounted_rewards(reward_history)

        total_reward = sum(reward_history)
        episode_rewards.append(total_reward)

        moving_num, window = 20, 100
        if episode >= window-1:
            moving_avg = np.mean(episode_rewards[-window:])
            print(f"Pong-v0 episode {episode}, reward sum: {total_reward}, last {window} avg: {moving_avg:.2f}", flush=True)
            
            if moving_avg > moving_num:
                print(f"Stopping as the last {window}-episode moving average is greater than {moving_num}", flush=True)
                break
        else:
            print(f"Pong-v0 episode {episode}, reward sum: {total_reward}", flush=True)

        ### TRAINING STOP

        if total_reward == 21:  # Check for consecutive rewards of 21
            consecutive_21_rewards += 1
            if consecutive_21_rewards >= 50 and should_train == True:
                print("Stopping training as the reward has been 21 for 50 episodes in a row", flush=True)
                should_train = False  # Set flag to False

                # Check if folder exists, if not create it
                if not os.path.exists("saved_model"):
                    os.makedirs("saved_model")

                # Save the model
                model.save("saved_model/pong_model")
                
        else:
            consecutive_21_rewards = 0  # Reset the counter if the reward is not 21

        # Modify the weight adjustment to respect the should_train flag
        # print("Starting Model Training", flush=True)
        if should_train:
            adjust_weights(model, optimizer, obs_history, action_history, discounted_rewards)
        # print("Ending Model Training", flush=True)

        # Save the model every 100 iterations
        if episode % 100 == 0:
            # model.save(f"saved_model/pong_model_{episode}")
            pass
            
        # Save a reward plot every 10 iterations with the same name
        if episode % 10 == 0:
            save_plot(episode_rewards, "latest")

        episode += 1

    env.close()

    # Save Results
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    # Calculate moving average of rewards
    window = 100  # Size of the window for calculating moving average
    moving_avgs = [np.mean(episode_rewards[max(0, i - window + 1):i+1]) for i in range(len(episode_rewards))]

    plt.plot(episode_rewards, label='Total Reward')
    plt.plot(moving_avgs, label=f'{window}-Episode Moving Average')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig('artifacts/pong_rewards.png')

if __name__ == "__main__":
    check_gpu_availability()
    Pong_RL()
    