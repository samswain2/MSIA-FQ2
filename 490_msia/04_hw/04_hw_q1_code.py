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

def set_seed(seed, use_cuda=True):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        self.population_ages = [0] * self.population_size
        self.fitness_scores = [0] * self.population_size
        self.replacement_proportion = config['genetic_algorithm']['replacement_proportion']

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

    def _binary_batch_size(self, minimum=None, maximum=None):
        min_bs = minimum if minimum is not None else self.min_batch_size
        max_bs = min(maximum, self.max_batch_size) if maximum is not None else self.max_batch_size
        batch_size = random.randint(min_bs, max_bs)
        binary_length = len(bin(self.max_batch_size)[2:])

        logging.debug(f"Generated binary len: {binary_length} for decimal batch size: {batch_size}")
    
        return [int(bit) for bit in bin(batch_size)[2:].zfill(binary_length)]
    
    def _initialize_population(self):
        population = []
        num_activation_funcs = len(self.activation_funcs)
        for _ in range(self.population_size):
            binary_batch_size = self._binary_batch_size()
            one_hot_activation_func = [0] * num_activation_funcs
            random_index = random.randint(0, num_activation_funcs - 1)
            one_hot_activation_func[random_index] = 1
            individual = binary_batch_size + one_hot_activation_func
            population.append(individual)
            logging.info(f"Initialized Individual: {individual}")
        return population
    
    def is_valid_one_hot_encoding(self, one_hot_vector):
        return sum(one_hot_vector) == 1 and 1 in one_hot_vector

    def is_valid_batch_size(self, binary_batch_size):
        batch_size = int(''.join(str(bit) for bit in binary_batch_size), 2)
        return self.min_batch_size <= batch_size <= self.max_batch_size

    def is_valid_individual(self, individual):
        return self.is_valid_one_hot_encoding(individual[-3:])

    def generate_random_individual(self):
        binary_batch_size = self._binary_batch_size()
        one_hot_activation_func = [0] * len(self.activation_funcs)
        random_index = random.randint(0, len(self.activation_funcs) - 1)
        one_hot_activation_func[random_index] = 1
        return binary_batch_size + one_hot_activation_func

    def adjust_batch_size(self, binary_batch_size):
        batch_size = int(''.join(str(bit) for bit in binary_batch_size), 2)
        if batch_size < self.min_batch_size:
            return self._binary_batch_size(minimum=self.min_batch_size)
        elif batch_size > self.max_batch_size:
            return self._binary_batch_size(maximum=self.max_batch_size)
        return binary_batch_size

    def calculate_fitness(self, individual):
        binary_batch_size = individual[:-3]
        one_hot_activation_func = individual[-3:]
        logging.debug(f"Binary Batch Size: {binary_batch_size}")
        logging.debug(f"One-Hot Activation Function: {one_hot_activation_func}")
        batch_size = int(''.join(str(bit) for bit in binary_batch_size), 2)
        logging.debug(f"Batch Size: {batch_size}")
        logging.debug(f"Activation Function OHE: {one_hot_activation_func}")
        activation_func_index = one_hot_activation_func.index(1)
        activation_func = self.activation_funcs[activation_func_index]
        logging.debug(f"Binary Batch Size: {binary_batch_size}, Converted Batch Size: {batch_size}")
        model = TwoLayerNet(self.X_train.shape[1], 128, 10)
        self.train_model(model, individual[:-3], activation_func)
        X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(self.y_val, dtype=torch.int64)
        outputs = model(X_val_tensor, activation_func)
        predicted = torch.argmax(outputs, 1)
        f1 = f1_score(y_val_tensor.numpy(), predicted.numpy(), average='macro')
        return f1
    
    def select_parents(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        selected_indices = np.random.choice(range(len(self.population)), size=self.population_size, p=selection_probs, replace=True)
        logging.debug(f"Selected Parent Indices: {selected_indices}")
        return [self.population[i] for i in selected_indices]
    
    def crossover(self, parent1, parent2):
        # Single crossover point for the entire individual
        crossover_point = random.randint(1, len(parent1))

        # Crossover
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        # Validate and adjust the children
        for child in [child1, child2]:
            batch_part = child[:-len(self.activation_funcs)]
            activation_part = child[-len(self.activation_funcs):]

            if not self.is_valid_batch_size(batch_part):
                batch_part = self.adjust_batch_size(batch_part)
            if not self.is_valid_one_hot_encoding(activation_part):
                activation_part = random.choice([[1] + [0] * (len(self.activation_funcs) - 1) for _ in range(len(self.activation_funcs))])

            child[:] = batch_part + activation_part

        return child1, child2
    
    def mutate(self, child):
        original_child = child.copy()
        if random.random() < self.mutation_rate:
            mutation_point = random.randint(0, len(child) - 1)
            child[mutation_point] = 1 - child[mutation_point]  # Flip bit

            # Adjust batch size if it goes out of range after mutation
            binary_batch_size = child[:-3]
            if not self.is_valid_batch_size(binary_batch_size):
                child = self.adjust_batch_size(binary_batch_size) + child[-3:]

            # Adjust activation function if it is not valid after mutation
            one_hot_activation_func = child[-3:]
            if not self.is_valid_one_hot_encoding(one_hot_activation_func):
                child[-3:] = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        logging.debug(f"Mutated Child: {original_child} -> {child}")
        return child
    
    def train_model(self, model, batch_size_binary, activation_func):
        batch_size = int(''.join(str(bit) for bit in batch_size_binary), 2)
        logging.debug(f"Training Model: Batch Size - {batch_size}, Activation Func - {activation_func.__name__}")
        
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.int64)
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
        batch_size = int(''.join(str(bit) for bit in best_individual[:-3]), 2)
        activation_func_index = best_individual[-3:].index(1)
        activation_func_name = self.activation_funcs[activation_func_index].__name__

        with open(file_path, 'w') as file:
            file.write(f"Generation: {generation}\n")
            file.write(f"Batch Size: {batch_size}\n")
            file.write(f"Activation Function: {activation_func_name}\n")

    def run(self):
        logging.info("Starting Genetic Algorithm")
        for generation in range(self.num_generations):
            logging.info(f"Generation {generation+1}/{self.num_generations}")

            # Calculate fitness for the entire population only in the first generation
            if generation == 0:
                self.fitness_scores = [self.calculate_fitness(individual) for individual in self.population]

            # Update average and best fitness scores
            average_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
            best_fitness = max(self.fitness_scores)
            self.average_fitness_scores.append(average_fitness)
            self.highest_fitness_scores.append(best_fitness)

            # Select parents based on fitness scores
            parents = self.select_parents(self.fitness_scores)

            # Generate a proportion of the population as children
            num_children_to_generate = int(self.population_size * self.replacement_proportion)
            children = []
            while len(children) < num_children_to_generate:
                for i in range(0, len(parents), 2):
                    if len(children) >= num_children_to_generate:
                        break
                    child1, child2 = self.crossover(parents[i], parents[min(i+1, len(parents)-1)])
                    child1 = self.adjust_batch_size(child1[:-3]) + child1[-3:]
                    child2 = self.adjust_batch_size(child2[:-3]) + child2[-3:]
                    children.extend([child1, child2])

            # Mutate children and calculate their fitness
            mutated_children = [self.mutate(child) for child in children]
            mutated_fitness = [self.calculate_fitness(child) for child in mutated_children]

            # Neatly log mutated children and population
            for i, child in enumerate(mutated_children):
                logging.debug(f"Mutated Child {i+1}: {child}")
            for i, individual in enumerate(self.population):
                logging.debug(f"Individual {i+1}: {individual}")

            # Log age of each chromosome
            for i, age in enumerate(self.population_ages):
                logging.debug(f"Age of Individual {i+1}: {age}")

            # Log fitness of each chromosome
            for i, fitness in enumerate(self.fitness_scores):
                logging.debug(f"Fitness of Individual {i+1}: {fitness}")

            # Replace the oldest chromosomes
            age_sorted_indices = np.argsort(self.population_ages)
            oldest_indices = age_sorted_indices[-len(mutated_children):]
            if len(set(self.population_ages)) < len(mutated_children):
                random.shuffle(oldest_indices)  # Random tie-breaking

            for index, (child, fitness) in zip(oldest_indices, zip(mutated_children, mutated_fitness)):
                self.population[index] = child
                self.fitness_scores[index] = fitness  # Update fitness score
                self.population_ages[index] = 0  # Reset age

            # Increment age of each chromosome
            self.population_ages = [age + 1 for age in self.population_ages]

            # Optionally, log or print the best fitness score in this generation
            logging.info(f"Average Fitness in Generation {generation+1}: {round(average_fitness, 5)}")
            logging.info(f"Best Fitness in Generation {generation+1}: {round(best_fitness, 5)}")
            logging.info("#" * 50)

            # Save plots and hyperparameters periodically
            if (generation + 1) % self.save_interval == 0:
                self.save_plot(generation + 1)
                best_individual_index = self.fitness_scores.index(max(self.fitness_scores))
                best_individual = self.population[best_individual_index]
                self.save_hyperparameters(best_individual, generation + 1)

        logging.info("Genetic Algorithm run completed")


class DataHandler:
    def __init__(self, train_X_path, train_y_path):
        self.train_X_path = train_X_path
        self.train_y_path = train_y_path

    def load_data(self):
        X_train = np.load(self.train_X_path)
        y_train = np.load(self.train_y_path)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)

        return X_train, y_train, X_val, y_val
    
# Main execution
def main():
    with open('04_hw_q1_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set random seed
    if 'random_seed' in config:
        set_seed(config['random_seed'])

    # Set GPU (if specified in config)
    if 'gpu_id' in config:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_id'])

    # Use the configuration for paths
    data_handler = DataHandler(config['data_paths']['train_X'], config['data_paths']['train_y'])
    X_train, y_train, X_val, y_val = data_handler.load_data()

    # Use the configuration for Genetic Algorithm
    ga = GeneticAlgorithm(X_train, y_train, X_val, y_val, config)
    ga.run()



if __name__ == "__main__":
    main()