import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import os

class FederatedLearningManager:
    def __init__(self, config):
        self.config = config
        self.train_data = np.load(config['train_data_path'], allow_pickle=True)
        test_data = np.load(config['test_data_path'], allow_pickle=True)
        self.test_images = np.array(test_data[0]['images'])
        self.test_labels = np.array(test_data[0]['labels'])
        self.client_train_val_data = self.prepare_client_data()
        self.global_model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(62, activation='softmax')  # There are 62 classes
        ])
        model.compile(optimizer='adam',
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
        model.fit(X_train, y_train, epochs=self.config['local_epochs'], batch_size=32, verbose=0)
        return model.get_weights()

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
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, round + 2), history['val_accuracy'], label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
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
        history = {'val_accuracy': [], 'val_loss': []}
        for round in range(self.config['communication_rounds']):
            selected_clients_indices = self.select_clients()
            weights_list = []
            for client_id in selected_clients_indices:
                local_model = self.create_model()
                local_model.set_weights(self.global_model.get_weights())
                client_weights = self.client_update(local_model, self.client_train_val_data[client_id])
                weights_list.append(client_weights)
            updated_weights = self.average_weights(weights_list)
            self.global_model.set_weights(updated_weights)
            val_loss, val_accuracy = self.global_model.evaluate(self.test_images, self.test_labels, verbose=0)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            print(f'Round {round+1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
            
            # Save the model and plot at specified intervals
            if (round + 1) % self.config['save_interval'] == 0:
                self.save_model(round)
                self.plot_progress(history, round)
                
        return self.global_model, history

# Configuration settings
config = {
    'train_data_path': "./Assignment3-data/train_data.npy",
    'test_data_path': "./Assignment3-data/test_data.npy",
    'communication_rounds': 200,
    'client_fraction': 0.1,
    'local_epochs': 5,
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
