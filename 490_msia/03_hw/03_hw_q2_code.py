import numpy as np
import random
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import ray
import logging
import warnings

@ray.remote(num_cpus=1, num_gpus=0.4)
class ClientActor:
    def __init__(self, client_data, model_config):
        self.client_data = client_data
        self.model = self.create_model(model_config)

    def create_model(self, model_config):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=model_config['input_shape']),
            tf.keras.layers.Dense(model_config['hidden_units'], activation='relu'),
            tf.keras.layers.Dense(model_config['output_units'], activation='softmax')
        ])
        optimizer = tf.keras.optimizers.SGD(learning_rate=model_config['learning_rate'])
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def train(self, model_weights, epochs):
        self.model.set_weights(model_weights)
        X_train = self.client_data['X_train']
        y_train = self.client_data['y_train']
        X_val = self.client_data['X_val']
        y_val = self.client_data['y_val']
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_val, y_val))
        weights = self.model.get_weights()
        val_loss, val_accuracy = history.history['val_loss'][-1], history.history['val_accuracy'][-1]
        train_loss, train_accuracy = history.history['loss'][-1], history.history['accuracy'][-1]
        return weights, val_loss, val_accuracy, train_loss, train_accuracy

class FederatedLearningManager:
    def __init__(self, config):
        ray.init(logging_level=logging.WARNING)
        self.config = config
        self.model_config = config['model_config']
        self.train_data = np.load(config['train_data_path'], allow_pickle=True)
        self.client_train_val_data = self.prepare_client_data()
        self.global_model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(62, activation='softmax')
        ])
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.model_config['learning_rate'])
        model.compile(optimizer=optimizer,
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
    
    @staticmethod
    def average_weights(weights_list):
        average_weights = []
        num_layers = len(weights_list[0])
        for layer in range(num_layers):
            layer_weights = np.array([client_weights[layer] for client_weights in weights_list])
            layer_average = np.mean(layer_weights, axis=0)
            average_weights.append(layer_average)
        return average_weights

    def save_model(self, iteration):
        # Convert the number of clients to a fraction string for directory naming
        # Assuming 100 total clients, as per your indication
        client_fraction_str = self.config['client_fraction']
        model_dir = os.path.join(self.config['model_save_dir'], f'fraction_{client_fraction_str}')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'global_model_round_{iteration+1}.h5')
        self.global_model.save(model_path)
        print(f'Model saved at {model_path}')

    def plot_progress(self, history, iteration):
        plt.figure(figsize=(12, 10))  # Adjust the figure size to accommodate 4 subplots

        # Plot Training Accuracy
        plt.subplot(2, 2, 1)
        plt.plot(range(1, iteration + 2), history['train_accuracy'], label='Training Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot Training Loss
        plt.subplot(2, 2, 2)
        plt.plot(range(1, iteration + 2), history['train_loss'], label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Rounds')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Validation Accuracy
        plt.subplot(2, 2, 3)
        plt.plot(range(1, iteration + 2), history['val_accuracy'], label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot Validation Loss
        plt.subplot(2, 2, 4)
        plt.plot(range(1, iteration + 2), history['val_loss'], label='Validation Loss')
        plt.title('Validation Loss')
        plt.xlabel('Rounds')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()  # Adjust subplots to fit into the figure area.
        plot_path = os.path.join(self.config['plot_save_dir'], f'training_progress_round_{iteration+1}.png')
        plt.savefig(plot_path)
        print(f'Progress plot saved at {plot_path}')
        plt.close()

    def federated_training(self):
        history = {'train_accuracy': [], 'train_loss': [], 'val_accuracy': [], 'val_loss': []}

        for iteration in range(self.config['communication_rounds']):
            selected_clients_indices = self.select_clients()

            # Initialize list to collect weights and validation metrics from all clients in the round
            all_client_weights = []
            client_train_metrics = []
            client_val_metrics = []

            # Process clients in batches due to GPU limitations
            batch_size = 4  # Number of clients processed in parallel, adjust as needed
            for i in range(0, len(selected_clients_indices), batch_size):
                batch_indices = selected_clients_indices[i:i + batch_size]
                client_actors = [ClientActor.remote(self.client_train_val_data[j], self.model_config) for j in batch_indices]

                # Perform client updates in parallel
                futures = [actor.train.remote(self.global_model.get_weights(), self.config['local_epochs']) for actor in client_actors]
                results = ray.get(futures)

                # Extend the weights list and collect validation metrics
                for weights, val_loss, val_accuracy, train_loss, train_accuracy in results:
                    all_client_weights.append(weights)
                    client_train_metrics.append((train_loss, train_accuracy))
                    client_val_metrics.append((val_loss, val_accuracy))

            # Average the weights from all clients and update the global model
            updated_weights = self.average_weights(all_client_weights)
            self.global_model.set_weights(updated_weights)

            # Calculate average training loss and accuracy across all clients
            avg_train_loss, avg_train_accuracy = np.mean(client_train_metrics, axis=0)
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(avg_train_accuracy)

            # Calculate average validation loss and accuracy across all clients
            avg_val_loss, avg_val_accuracy = np.mean(client_val_metrics, axis=0)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(avg_val_accuracy)
            print(f"""Round {iteration+1}, Train Loss: {round(avg_train_loss, 4)}, Train Accuracy: {round(avg_train_accuracy, 4)}, Validation Loss: {round(avg_val_loss, 4)}, Validation Accuracy: {round(avg_val_accuracy, 4)}""")

            # Save the model and plot at specified intervals
            if (iteration + 1) % self.config['save_interval'] == 0:
                self.save_model(iteration)
                self.plot_progress(history, iteration)

        return self.global_model, history

# Configuration settings
config = {
    'train_data_path': "./Assignment3-data/train_data.npy",
    'communication_rounds': 1000,
    'client_fraction': 0.04,
    'local_epochs': 20,
    'save_interval': 50,  # Save model and plot every 5 rounds
    'model_save_dir': './saved_models_parallel',  # Directory to save models
    'plot_save_dir': './training_plots_parallel',  # Directory to save plots
    'model_config': {  # Model configuration
        'input_shape': (28, 28),
        'hidden_units': 128,
        'output_units': 62,
        'learning_rate': 0.001
    },
    'gpu_id': '0,1',  # Set this to the desired GPU ID
}

# Set the GPU ID from the config and disable logs
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Ensure directories exist
os.makedirs(config['model_save_dir'], exist_ok=True)
os.makedirs(config['plot_save_dir'], exist_ok=True)

# Initialize and run federated learning
fl_manager = FederatedLearningManager(config)
global_model, training_history = fl_manager.federated_training()
