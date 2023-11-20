import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

class FederatedLearningManager:
    def __init__(self, config):
        self.config = config
        self.set_gpu(self.config['gpu_id'])
        self.train_data = np.load(config['train_data_path'], allow_pickle=True)
        self.client_train_val_data = self.prepare_client_data()
        self.global_model = self.create_model()

    def set_gpu(self, gpu_id):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Restrict TensorFlow to only use the specified GPU
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(62, activation='softmax')  # Output layer for 62 classes
        ])

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.config['learning_rate']),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def prepare_client_data(self):
        client_data = {}
        for client_id in range(len(self.train_data)):
            X_train, X_val, y_train, y_val = self.split_data(self.train_data[client_id])
            client_data[client_id] = {
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val
            }
        return client_data

    @staticmethod
    def split_data(client_data):
        images = np.array(client_data['images'])
        labels = np.array(client_data['labels'])
        return train_test_split(images, labels, test_size=0.2, random_state=42)

    def select_clients(self):
        num_clients = len(self.train_data)
        fraction = self.config['client_fraction']
        num_selected = max(int(num_clients * fraction), 1)
        return random.sample(range(num_clients), num_selected)

    def client_update(self, model, client_data):
        X_train = np.array(client_data['X_train'])
        y_train = np.array(client_data['y_train'])
        X_val = np.array(client_data['X_val'])
        y_val = np.array(client_data['y_val'])

        history = model.fit(X_train, y_train, epochs=self.config['local_epochs'], batch_size=32, validation_data=(X_val, y_val), verbose=0)

        train_loss, train_accuracy = history.history['loss'][-1], history.history['accuracy'][-1]
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

        return model.get_weights(), train_loss, train_accuracy, val_loss, val_accuracy, len(X_train)

    @staticmethod
    def average_weights(weights_list, sample_sizes):
        average_weights = []
        num_layers = len(weights_list[0])

        # Calculate the total number of samples
        total_samples = sum(sample_sizes)

        for layer in range(num_layers):
            weighted_layer_sum = np.sum(
                np.array([client_weights[layer] * sample_sizes[i] for i, client_weights in enumerate(weights_list)]), 
                axis=0
            )
            weighted_layer_average = weighted_layer_sum / total_samples
            average_weights.append(weighted_layer_average)
        
        return average_weights

    def save_model(self, iteration):
        model_path = os.path.join(self.config['model_save_dir'], f'global_model_round_{iteration+1}.h5')
        self.global_model.save(model_path)
        print(f'Model saved at {model_path}')

    def plot_progress(self, history, iteration, client_fraction):
        plt.figure(figsize=(12, 10))
        plt.suptitle(f'Client Fraction: {client_fraction}', fontsize=14)

        plt.subplot(2, 2, 1)
        plt.plot(range(1, iteration + 2), history['train_accuracy'], label='Train Accuracy')
        plt.title('Train Accuracy')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(range(1, iteration + 2), history['train_loss'], label='Train Loss')
        plt.title('Train Loss')
        plt.xlabel('Rounds')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(range(1, iteration + 2), history['val_accuracy'], label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(range(1, iteration + 2), history['val_loss'], label='Validation Loss')
        plt.title('Validation Loss')
        plt.xlabel('Rounds')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.config['plot_save_dir'], f'training_progress_round_{iteration+1}.png')
        plt.savefig(plot_path)
        print(f'Progress plot saved at {plot_path}')
        plt.close()

    def federated_training(self):
        global_history = {
            'train_accuracy': [], 'train_loss': [],
            'val_accuracy': [], 'val_loss': []
        }

        for iteration in range(self.config['communication_rounds']):
            selected_clients_indices = self.select_clients()
            weights_list = []
            sample_sizes = []
            round_metrics = {'train_accuracy': 0, 'train_loss': 0, 'val_accuracy': 0, 'val_loss': 0, 'total_samples': 0}

            for client_id in selected_clients_indices:
                local_model = self.create_model()
                local_model.set_weights(self.global_model.get_weights())
                client_weights, train_loss, train_accuracy, val_loss, val_accuracy, num_samples = self.client_update(local_model, self.client_train_val_data[client_id])

                weights_list.append(client_weights)
                sample_sizes.append(num_samples)
                round_metrics['train_accuracy'] += train_accuracy * num_samples
                round_metrics['train_loss'] += train_loss * num_samples
                round_metrics['val_accuracy'] += val_accuracy * num_samples
                round_metrics['val_loss'] += val_loss * num_samples
                round_metrics['total_samples'] += num_samples

            # Update global model with the average weights after all clients have been processed
            updated_weights = self.average_weights(weights_list, sample_sizes)
            self.global_model.set_weights(updated_weights)

            # Calculate weighted average of metrics for the round
            for metric in ['train_accuracy', 'train_loss', 'val_accuracy', 'val_loss']:
                global_history[metric].append(round_metrics[metric] / round_metrics['total_samples'])

            # Print round summary
            print(f'Round {iteration+1}, Train Loss: {round(global_history["train_loss"][-1], 4)}, '
                f'Train Accuracy: {round(global_history["train_accuracy"][-1], 4)}, '
                f'Validation Loss: {round(global_history["val_loss"][-1], 4)}, '
                f'Validation Accuracy: {round(global_history["val_accuracy"][-1], 4)}'
                )
                
            # Save the model and plot at specified intervals
            if (iteration + 1) % self.config['save_interval'] == 0:
                self.save_model(iteration)
                self.plot_progress(global_history, iteration, self.config['client_fraction'])
                
        return self.global_model, global_history

#  Configuration settings common to all runs
base_config = {
    'train_data_path': "./Assignment3-data/train_data.npy",
    'communication_rounds': 1000,
    'local_epochs': 20,
    'learning_rate': 0.001,  # Specify the learning rate
    'save_interval': 100,  # Save model and plot every 5 rounds
    'model_save_dir': './saved_models',  # Directory to save models
    'plot_save_dir': './training_plots',  # Directory to save plots
    'gpu_id': 0  # Specify which GPU to use
}

# Client fractions to try
client_fractions = [0.025, 0.05, 0.075, 0.1]

# Dictionary to store final results for each client fraction
final_results = {}

# Loop over each client fraction
for fraction in client_fractions:
    # Update client fraction in configuration
    config = base_config.copy()
    config['client_fraction'] = fraction

    # Update directories to include client fraction
    fraction_dir = f'fraction_{fraction}'
    config['model_save_dir'] = os.path.join('./saved_models', fraction_dir)
    config['plot_save_dir'] = os.path.join('./training_plots', fraction_dir)

    # Ensure directories exist
    os.makedirs(config['model_save_dir'], exist_ok=True)
    os.makedirs(config['plot_save_dir'], exist_ok=True)

    # Initialize and run federated learning
    fl_manager = FederatedLearningManager(config)
    global_model, training_history = fl_manager.federated_training()

    # Store the final results
    final_results[fraction] = {
        'model': global_model,
        'history': training_history
    }

# Output final results
for fraction, results in final_results.items():
    print(f"Results for client fraction {fraction}:")
    