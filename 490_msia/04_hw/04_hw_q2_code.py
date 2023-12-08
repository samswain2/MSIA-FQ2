import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from functools import partial
import matplotlib.pyplot as plt
import yaml

# Load configurations from YAML file
with open('04_hw_q2_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set device
device = torch.device(config['device'])

# Load data
train_X = np.load(config['data']['train_X_path'])
train_y = np.load(config['data']['train_y_path'])

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_X, train_y, 
    test_size=config['data_split']['test_size'], 
    random_state=config['data_split']['random_state']
)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, device=device)
y_train = torch.tensor(y_train, dtype=torch.long, device=device)
X_val = torch.tensor(X_val, device=device)
y_val = torch.tensor(y_val, dtype=torch.long, device=device)

# Neural network model
class Net(nn.Module):
    def __init__(self, activation_fn):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input images are 28x28
        self.fc2 = nn.Linear(128, 10)
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_fn == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to calculate F1 score
def f1_score_pytorch(y_true, y_pred):
    y_true = y_true.cpu()
    return f1_score(y_true, np.argmax(y_pred, axis=1), average='macro')

# Training function
def train(network, loader, optimizer):
    network.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

# Evaluation function
def validate(network, loader):
    network.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            # Move output to CPU before converting to NumPy
            output_cpu = output.cpu()
            return f1_score_pytorch(target, output_cpu.numpy())

# Black-box function for Bayesian optimization
def black_box_function(batch_size, activation_index, X_train, y_train, X_val, y_val):
    activation_fn = config['activation_functions'][int(activation_index)]
    network = Net(activation_fn).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=int(batch_size), shuffle=False)

    optimizer = optim.SGD(network.parameters(), lr=config['optimizer']['lr'], momentum=config['optimizer']['momentum'])

    # Training for a fixed number of epochs
    for epoch in range(config['training']['epochs']):
        train(network, train_loader, optimizer)
    
    f1 = validate(network, val_loader)
    return f1

# Function to save plot of the Bayesian optimization process
def save_bo_plot(optimizer, plot_save_path):
    os.makedirs(plot_save_path, exist_ok=True)
    target_scores = [res["target"] for res in optimizer.res]
    plt.figure()
    plt.plot(target_scores, label='BO Score')
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.title('Bayesian Optimization Process')
    plt.legend()
    plt.savefig(os.path.join(plot_save_path, "bo_process.png"))
    plt.close()

# Function to save the best hyperparameters found by Bayesian optimization
def save_best_hyperparameters(best_params, hyperparameters_save_path, activation_functions):
    os.makedirs(hyperparameters_save_path, exist_ok=True)
    file_path = os.path.join(hyperparameters_save_path, "best_hyperparameters_BO.txt")

    # Round batch_size to the nearest integer
    batch_size = int(round(best_params['batch_size']))

    # Map activation_index to the corresponding activation function name
    # Ensure that the index is within the valid range
    activation_index = int(round(best_params['activation_index']))
    activation_index = max(0, min(activation_index, len(activation_functions) - 1))
    activation_func_name = activation_functions[activation_index]

    with open(file_path, 'w') as file:
        file.write(f"Batch Size: {batch_size}\n")
        file.write(f"Activation Function: {activation_func_name}\n")

# Wrap the black-box function
func = partial(black_box_function, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

# Bayesian optimization
optimizer = BayesianOptimization(
    f=func,
    pbounds={'batch_size': tuple(config['bayesian_opt']['batch_size']), 'activation_index': tuple(config['bayesian_opt']['activation_index'])},
    random_state=config['bayesian_opt']['random_state'],
    verbose=config['bayesian_opt']['verbose']
)
optimizer.maximize(init_points=config['bayesian_opt']['init_points'], n_iter=config['bayesian_opt']['n_iter'])

# Retrieve the best parameters
best_params = optimizer.max['params']
print("Best parameters found by Bayesian Optimization: ", best_params)

# Save the optimization plot and best hyperparameters
plot_save_path = os.path.join('artifacts', 'plots')
hyperparameters_save_path = os.path.join('artifacts', 'hyperparameters')
save_bo_plot(optimizer, plot_save_path)
save_best_hyperparameters(best_params, hyperparameters_save_path, config['activation_functions'])
