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

def select_action(model, observation):
    action_probs, _ = model(tf.expand_dims(observation, axis=0))
    action_probs = action_probs[0].numpy()
    action_idx = np.random.choice([0, 1], p=action_probs)
    return action_idx

def save_plot(episode_rewards):
    window = 100  # Size of the window for calculating moving average
    moving_avgs = [np.mean(episode_rewards[max(0, i - window + 1):i + 1]) for i in range(len(episode_rewards))]
    
    plt.plot(episode_rewards, label='Total Reward')
    plt.plot(moving_avgs, label=f'{window}-Episode Moving Average')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig(f'artifacts/500_pong_rollout.png')
    plt.close()

def save_histogram(episode_rewards):
    plt.hist(episode_rewards, bins=20, edgecolor='black')
    plt.xlabel('Episode Rewards')
    plt.ylabel('Frequency')
    plt.title('Histogram of Episode Rewards')
    plt.savefig('artifacts/episode_rewards_histogram.png')
    plt.close()

def Pong_RL():
    env = gym.make("Pong-v0", 
                #    render_mode="human"
                   )

    ### Code below will load existing model in
    model_path = f"saved_model/pong_model_submission"  # Adjust this to your specific model path

    # Attempt to load the model
    if os.path.exists(model_path):
        print("Loading existing model from", model_path, flush=True)
        model = tf.keras.models.load_model(model_path)
    else:
        print("NO MODEL WELP")

    model.summary()
    episode_rewards = []

    # Game
    num_episodes = 500
    for episode in range(1, num_episodes + 1):
        observation, info = env.reset()
        obs_history, reward_history, action_history = [], [], []

        terminated = False
        truncated = False

        while not terminated and not truncated:
            processed_observation = preprocess(observation)
            action = select_action(model, processed_observation)
            obs_history.append(processed_observation)
            action_history.append(action)

            action_to_take = 2 if action == 0 else 3  # Remap the action index to the appropriate Pong-v0 action
            observation, reward, terminated, truncated, info = env.step(action_to_take)
            
            reward_history.append(reward)

        total_reward = sum(reward_history)
        episode_rewards.append(total_reward)
        
        print(f"Episode {episode} of {num_episodes}: Total Reward = {total_reward}")
        
        episode += 1

    env.close()

    # Save Results
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    save_plot(episode_rewards)
    save_histogram(episode_rewards)

    print(f"Mean: {np.mean(episode_rewards)}")
    print(f"Standard Dev.: {np.std(episode_rewards)}")

    # Added code to output mean and std to text file
    with open('artifacts/500_rollout_results.txt', 'w') as f:
        f.write(f"Mean: {np.mean(episode_rewards)}\n")
        f.write(f"Standard Dev.: {np.std(episode_rewards)}\n")

    np.save('artifacts/500_rollout_results_array.npy', np.array(episode_rewards))

if __name__ == "__main__":
    check_gpu_availability()
    Pong_RL()
    