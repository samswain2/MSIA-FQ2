import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import yaml

# Load configurations from YAML file
with open('04_hw_test_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define your neural network architecture
class Net(nn.Module):
    def __init__(self, activation_fn):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Assuming input images are 28x28
        self.fc2 = nn.Linear(128, 10)
        self.activation = activation_fn

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the training function
def train(network, loader, optimizer):
    network.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

    # Return f1 score
    return validate(network, loader)

# Define the evaluation function
def validate(network, loader):
    network.eval()
    total_f1 = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = network(data)
        pred = output.argmax(dim=1, keepdim=True)
        total_f1 += f1_score_pytorch(target, pred.squeeze())
    return total_f1 / len(loader)

# Function to calculate F1 score
def f1_score_pytorch(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return f1_score(y_true, y_pred, average='macro')

# Function to read hyperparameters from a file
def read_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        hyperparams = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':')
                hyperparams[key.strip()] = value.strip()
    return hyperparams

# Function to convert activation function name to the actual function
def get_activation_function(name):
    if name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif name.lower() == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load full training data and test data
train_X = np.load(config['data']['train_X_path'])
train_y = np.load(config['data']['train_y_path'])
test_X = np.load(config['data']['test_X_path'])
test_y = np.load(config['data']['test_y_path'])

# Convert data to PyTorch tensors
X_train = torch.tensor(train_X, dtype=torch.float32)
y_train = torch.tensor(train_y, dtype=torch.int64)
X_test = torch.tensor(test_X, dtype=torch.float32)
y_test = torch.tensor(test_y, dtype=torch.int64)

# Read hyperparameters from files
ga_hyperparams = read_hyperparameters('artifacts/hyperparameters/best_hyperparameters_gen_200.txt')
bo_hyperparams = read_hyperparameters('artifacts/hyperparameters/best_hyperparameters_BO.txt')

# Prepare models with these hyperparameters
ga_model = Net(get_activation_function(ga_hyperparams['Activation Function'])).to(device)
bo_model = Net(get_activation_function(bo_hyperparams['Activation Function'])).to(device)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=max(int(ga_hyperparams['Batch Size']), int(bo_hyperparams['Batch Size'])), shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)

# Train GA model
ga_f1_history = []
ga_optimizer = optim.SGD(ga_model.parameters(), lr=config['optimizer']['lr'], momentum=config['optimizer']['momentum'])
for epoch in range(config['final_training']['epochs']):
    ga_f1_temp = train(ga_model, train_loader, ga_optimizer)
    print(f"GA model f1 score after epoch {epoch+1}: {ga_f1_temp}")
    ga_f1_history.append(ga_f1_temp)

# Train BO model
bo_f1_history = []
bo_optimizer = optim.SGD(bo_model.parameters(), lr=config['optimizer']['lr'], momentum=config['optimizer']['momentum'])
for epoch in range(config['final_training']['epochs']):
    bo_f1_temp = train(bo_model, train_loader, bo_optimizer)
    print(f"BO model f1 score after epoch {epoch+1}: {bo_f1_temp}")
    bo_f1_history.append(bo_f1_temp)

# Evaluate models
ga_f1 = validate(ga_model, test_loader)
bo_f1 = validate(bo_model, test_loader)

# Plot GA and BO training history
plt.plot(range(1, len(ga_f1_history) + 1), ga_f1_history, label='GA Model')
plt.plot(range(1, len(bo_f1_history) + 1), bo_f1_history, label='BO Model')

# Set plot labels and title
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Training History')

# Add legend
plt.legend()

# Save the plot
plt.savefig('artifacts/training_history.png')

# Output the results to a text file
results_path = 'artifacts/comparison_results.txt'
with open(results_path, 'w') as file:
    file.write("GA Model Results:\n")
    file.write(f"Batch Size: {ga_hyperparams['Batch Size']}\n")
    file.write(f"Activation Function: {ga_hyperparams['Activation Function']}\n")
    file.write(f"Test F1 Score: {ga_f1}\n\n")

    file.write("BO Model Results:\n")
    file.write(f"Batch Size: {bo_hyperparams['Batch Size']}\n")
    file.write(f"Activation Function: {bo_hyperparams['Activation Function']}\n")
    file.write(f"Test F1 Score: {bo_f1}\n\n")

    # Write a simple comparison
    better_model = "GA" if ga_f1 > bo_f1 else "BO"
    file.write(f"The {better_model} model performed better on the test data.\n")

print(f"Comparison results have been written to {results_path}")
