# Use validation sets to evaluate on each client, then send back the train loss, acc, val loss, acc, and len of sample

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import os
from sklearn.utils.class_weight import compute_class_weight

class FederatedLearningManager:
    def __init__(self, config):
        self.config = config
        self.train_data = np.load(config['train_data_path'], allow_pickle=True)
        self.test_data = np.load(config['test_data_path'], allow_pickle=True)
        self.X_test = np.array(self.test_data[0]['images'])
        self.y_test = np.array(self.test_data[0]['labels'])
        self.client_train_val_data = self.prepare_client_data()
        self.global_model = self.create_model()

    def create_model(self):
        learning_rate = 0.001

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.Dense(62, activation='softmax')  # Output layer for 62 classes
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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
    def average_weights(weights_list):
        average_weights = []
        num_layers = len(weights_list[0])
        for layer in range(num_layers):
            layer_weights = np.array([client_weights[layer] for client_weights in weights_list])
            layer_average = np.mean(layer_weights, axis=0)
            average_weights.append(layer_average)
        return average_weights

    def save_model(self, round):
        model_path = os.path.join(self.config['model_save_dir'], f'global_model_round_{round+1}.h5')
        self.global_model.save(model_path)
        print(f'Model saved at {model_path}')

    def plot_progress(self, history, round):
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        plt.plot(range(1, round + 2), history['train_accuracy'], label='Train Accuracy')
        plt.title('Train Accuracy')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(range(1, round + 2), history['train_loss'], label='Train Loss')
        plt.title('Train Loss')
        plt.xlabel('Rounds')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(range(1, round + 2), history['val_accuracy'], label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(range(1, round + 2), history['val_loss'], label='Validation Loss')
        plt.title('Validation Loss')
        plt.xlabel('Rounds')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.config['plot_save_dir'], f'training_progress_round_{round+1}.png')
        plt.savefig(plot_path)
        print(f'Progress plot saved at {plot_path}')
        plt.close()

    def federated_training(self):
        global_history = {
            'train_accuracy': [], 'train_loss': [],
            'val_accuracy': [], 'val_loss': [],
            'test_accuracy': [], 'test_loss': []  # Added test metrics
        }

        for round in range(self.config['communication_rounds']):
            selected_clients_indices = self.select_clients()
            weights_list = []
            round_metrics = {'train_accuracy': 0, 'train_loss': 0, 'val_accuracy': 0, 'val_loss': 0, 'total_samples': 0}

            for client_id in selected_clients_indices:
                local_model = self.create_model()
                local_model.set_weights(self.global_model.get_weights())
                client_weights, train_loss, train_accuracy, val_loss, val_accuracy, num_samples = self.client_update(local_model, self.client_train_val_data[client_id])

                weights_list.append(client_weights)
                round_metrics['train_accuracy'] += train_accuracy * num_samples
                round_metrics['train_loss'] += train_loss * num_samples
                round_metrics['val_accuracy'] += val_accuracy * num_samples
                round_metrics['val_loss'] += val_loss * num_samples
                round_metrics['total_samples'] += num_samples

            # Update global model with the average weights after all clients have been processed
            updated_weights = self.average_weights(weights_list)
            self.global_model.set_weights(updated_weights)
            
            # Evaluate on test data after updating the global model
            test_loss, test_accuracy = self.global_model.evaluate(self.X_test, self.y_test, verbose=0)
            global_history['test_accuracy'].append(test_accuracy)
            global_history['test_loss'].append(test_loss)

            # Calculate weighted average of metrics for the round
            for metric in ['train_accuracy', 'train_loss', 'val_accuracy', 'val_loss']:
                global_history[metric].append(round_metrics[metric] / round_metrics['total_samples'])

            # Print round summary after test evaluation
            print(f'Round {round+1}, Train Loss: {global_history["train_loss"][-1]}, '
                f'Train Accuracy: {global_history["train_accuracy"][-1]}, '
                f'Validation Loss: {global_history["val_loss"][-1]}, '
                f'Validation Accuracy: {global_history["val_accuracy"][-1]}, '
                f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
            
            # Save the model and plot at specified intervals
            if (round + 1) % self.config['save_interval'] == 0:
                self.save_model(round)
                self.plot_progress(global_history, round)
                
        return self.global_model, global_history


# Configuration settings
config = {
    'train_data_path': "./Assignment3-data/train_data.npy",
    'test_data_path': "./Assignment3-data/test_data.npy",
    'communication_rounds': 500,
    'client_fraction': 0.1,
    'local_epochs': 20,
    'save_interval': 5,  # Save model and plot every 5 rounds
    'model_save_dir': './saved_models',  # Directory to save models
    'plot_save_dir': './training_plots'  # Directory to save plots
}

# Ensure directories exist
os.makedirs(config['model_save_dir'], exist_ok=True)
os.makedirs(config['plot_save_dir'], exist_ok=True)

# Initialize and run federated learning
fl_manager = FederatedLearningManager(config)
global_model, training_history = fl_manager.federated_training()
