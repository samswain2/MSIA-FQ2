import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import logging
import yaml
import matplotlib.pyplot as plt
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Neural Network Class
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, activation_func=F.relu):
        out = activation_func(self.fc1(x))
        out = self.fc2(out)
        return out
    
class GeneticAlgorithm:
    def __init__(self, X_train, y_train, X_val, y_val, config):
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.config = config
        self.population_size = config['genetic_algorithm']['population_size']
        self.num_generations = config['genetic_algorithm']['num_generations']
        self.mutation_rate = config['genetic_algorithm']['mutation_rate']

        # Define activation functions and batch sizes before initializing the population
        activation_func_mapping = {
            'relu': F.relu,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh
        }
        self.activation_funcs = [activation_func_mapping[func_name] for func_name in config['genetic_algorithm']['activation_funcs']]
        batch_size_range = config['genetic_algorithm']['batch_size_range']
        self.min_batch_size = batch_size_range[0]
        self.max_batch_size = batch_size_range[1]

        # Attributes for storing fitness data
        self.average_fitness_scores = []
        self.highest_fitness_scores = []
        self.save_interval = config['save_interval']
        self.plot_save_path = config['plot_save_path']
        self.model_save_path = config['model_save_path']
        self.hyperparameters_save_path = config['hyperparameters_save_path']

        # Now initialize the population
        self.population = self._initialize_population()

    def _random_batch_size(self):
        return random.randint(self.min_batch_size, self.max_batch_size)

    def _initialize_population(self):
        return [[self._random_batch_size(), random.choice(self.activation_funcs)] for _ in range(self.population_size)]

    def calculate_fitness(self, individual):
        model = TwoLayerNet(self.X_train.shape[1], 128, 10)
        self.train_model(model, individual[0], individual[1])
        X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(self.y_val, dtype=torch.int64)
        outputs = model(X_val_tensor, individual[1])
        predicted = torch.argmax(outputs, 1)
        f1 = f1_score(y_val_tensor.numpy(), predicted.numpy(), average='macro')
        return f1

    def select_parents(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        selected_indices = np.random.choice(range(len(self.population)), size=self.population_size, p=selection_probs, replace=True)
        logging.info(f"Selected Parent Indices: {selected_indices}")
        return [self.population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:], parent2[:crossover_point] + parent1[crossover_point:]

    def mutate(self, child):
        if random.random() < self.mutation_rate:
            gene_to_mutate = random.choice(['batch_size', 'activation_func'])
            if gene_to_mutate == 'batch_size':
                child[0] = self._random_batch_size()
            else:
                child[1] = random.choice(self.activation_funcs)
        return child

    def train_model(self, model, batch_size, activation_func):
        # Convert data to tensors
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.int64)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config['genetic_algorithm']['learning_rate'])

        # Training loop
        epochs = self.config['genetic_algorithm']['epochs']
        for epoch in range(epochs):
            for i in range(0, len(self.X_train), batch_size):
                X_batch = X_train_tensor[i:i+batch_size]
                y_batch = y_train_tensor[i:i+batch_size]

                # Forward pass
                outputs = model(X_batch, activation_func)
                loss = criterion(outputs, y_batch)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Logging the loss at each epoch
            # logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # def save_model(self, model, generation):
    #     os.makedirs(self.model_save_path, exist_ok=True)  # Ensure directory exists
    #     torch.save(model.state_dict(), f"{self.model_save_path}model_gen_{generation}.pt")

    def save_plot(self, generation):
        os.makedirs(self.plot_save_path, exist_ok=True)  # Ensure directory exists
        plt.figure()
        plt.plot(range(1, generation + 1), self.average_fitness_scores, label='Average Fitness')
        plt.plot(range(1, generation + 1), self.highest_fitness_scores, label='Highest Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.title(f'Fitness Scores up to Generation {generation}')
        plt.legend()
        plt.savefig(f"{self.plot_save_path}fitness_plot_gen_{generation}.png")
        plt.close()

    def save_hyperparameters(self, best_individual, generation):
        os.makedirs(self.hyperparameters_save_path, exist_ok=True)
        file_path = os.path.join(self.hyperparameters_save_path, f"best_hyperparameters_gen_{generation}.txt")
        with open(file_path, 'w') as file:
            file.write(f"Generation: {generation}\n")
            file.write(f"Batch Size: {best_individual[0]}\n")
            file.write(f"Activation Function: {best_individual[1].__name__}\n")

    def run(self):
        for generation in range(self.num_generations):
            logging.info(f"Generation {generation+1}/{self.num_generations}")
            fitness_scores = [self.calculate_fitness(individual) for individual in self.population]

            # Update fitness scores
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            best_fitness = max(fitness_scores)
            self.average_fitness_scores.append(average_fitness)
            self.highest_fitness_scores.append(best_fitness)

            parents = self.select_parents(fitness_scores)
            children = []
            while len(children) < self.population_size:
                for i in range(0, len(parents), 2):
                    child1, child2 = self.crossover(parents[i], parents[min(i+1, len(parents)-1)])
                    children.extend([child1, child2])
            children = children[:self.population_size]
            mutated_children = [self.mutate(child) for child in children]
            self.population = mutated_children

            if (generation + 1) % self.save_interval == 0:
                self.save_plot(generation + 1)
                best_individual_index = np.argmax([self.calculate_fitness(individual) for individual in self.population])
                best_individual = self.population[best_individual_index]
                best_model = TwoLayerNet(self.X_train.shape[1], 128, 10)
                self.train_model(best_model, best_individual[0], best_individual[1])
                # self.save_model(best_model, generation + 1)
                self.save_hyperparameters(best_individual, generation + 1)

            # Optionally, log or print the best fitness score in this generation
            best_fitness = max(fitness_scores)
            logging.info(f"Average Fitness in Generation {generation+1}: {round(average_fitness, 5)}")
            logging.info(f"Best Fitness in Generation {generation+1}: {round(best_fitness, 5)}")
            logging.info("#" * 50)


class DataHandler:
    def __init__(self, train_X_path, train_y_path, test_X_path, test_y_path):
        self.train_X_path = train_X_path
        self.train_y_path = train_y_path
        self.test_X_path = test_X_path
        self.test_y_path = test_y_path

    def load_data(self):
        X_train = np.load(self.train_X_path)
        y_train = np.load(self.train_y_path)
        X_test = np.load(self.test_X_path)
        y_test = np.load(self.test_y_path)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        return X_train, y_train, X_val, y_val, X_test, y_test
    
# Main execution
def main():
    with open('04_hw_q1_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Use the configuration for paths
    data_handler = DataHandler(config['data_paths']['train_X'], config['data_paths']['train_y'],
                               config['data_paths']['test_X'], config['data_paths']['test_y'])
    X_train, y_train, X_val, y_val, X_test, y_test = data_handler.load_data()

    # Use the configuration for Genetic Algorithm
    ga = GeneticAlgorithm(X_train, y_train, X_val, y_val, config)
    ga.run()



if __name__ == "__main__":
    main()