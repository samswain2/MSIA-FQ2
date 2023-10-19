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

        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set the second GPU as available
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


def build_model(input_shape=(80, 80, 1), num_choices=2, reg=0.0001):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),  # Set input_shape here
        tf.keras.layers.Dense(256, activation='relu'),  # Dense hidden layer
        tf.keras.layers.Dense(num_choices, activation="softmax")  # Dense output layer
    ])
    return model 


def select_action(model, observation):
    probabilities = model.predict(tf.expand_dims(observation, axis=0), verbose=0)[0]
    action = np.random.choice([2, 3], p=probabilities)  # 2 is RIGHT and 3 is LEFT
    return action

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

def adjust_weights(model, optimizer, obs_history, action_history, discounted_rewards):
    print("Getting discounted rewards")
    discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
    with tf.GradientTape() as tape:
        print("Getting probs")
        probs=model(tf.convert_to_tensor(obs_history, dtype=tf.float32))
        print("Getting Indices")
        indices=tf.stack([tf.range(len(action_history),dtype=tf.int32),tf.convert_to_tensor(action_history,dtype=tf.int32)],axis=1)
        print("Getting Chosen Probs")
        chosen_probs=tf.gather_nd(probs,indices)
        print("Getting Loss")
        loss=-tf.math.log(chosen_probs)*discounted_rewards
        loss=tf.reduce_mean(loss)
    print("Applying Gradient and Optimizer")
    grads=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))

def save_plot(episode_rewards, episode):
    window = 100  # Size of the window for calculating moving average
    moving_avgs = [np.mean(episode_rewards[max(0, i - window + 1):i + 1]) for i in range(len(episode_rewards))]
    
    plt.plot(episode_rewards, label='Total Reward')
    plt.plot(moving_avgs, label=f'{window}-Episode Moving Average')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig(f'artifacts/pong_rewards_{episode}.png')
    plt.close()

def Pong_RL():
    env = gym.make("Pong-v0")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = build_model()
    episode_rewards = []
    episode = 0 

    consecutive_21_rewards = 0  # Count number of 21 occurences
    should_train = True  # Initialize flag for training

    while True:
        observation, info = env.reset()
        obs_history, reward_history, action_history = [], [], []

        terminated = False
        truncated = False

        print("Starting Game")

        while not terminated and not truncated:
            processed_observation = preprocess(observation)
            action = select_action(model, processed_observation)
            obs_history.append(processed_observation)
            action_history.append(action)

            observation, reward, terminated, truncated, info = env.step(action)
            
            reward_history.append(reward)

        print("Finishing Game")

        discounted_rewards = compute_discounted_rewards(reward_history)

        total_reward = sum(reward_history)
        episode_rewards.append(total_reward)

        moving_num, window = 0, 100
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
            if consecutive_21_rewards >= 10 and should_train == True:
                print("Stopping training as the reward has been 21 for 10 episodes in a row", flush=True)
                should_train = False  # Set flag to False

                # Check if folder exists, if not create it
                if not os.path.exists("saved_model"):
                    os.makedirs("saved_model")

                # Save the model
                model.save("saved_model/pong_model")
                
        else:
            consecutive_21_rewards = 0  # Reset the counter if the reward is not 21

        # Modify the weight adjustment to respect the should_train flag
        print("Starting Model Training")
        if should_train:
            adjust_weights(model, optimizer, obs_history, action_history, discounted_rewards)
        print("Ending Model Training")

        # Save the model every 100 iterations
        if episode % 100 == 0:
            model.save(f"saved_model/pong_model_{episode}")
            
        # Save a reward plot every iteration
        save_plot(episode_rewards, episode)

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
    