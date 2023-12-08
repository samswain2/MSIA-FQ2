# AutoML Assignment on Federated EMNIST Dataset

### Overview
This project implements a 2-layer feed-forward neural network for image classification on a subset of the Federated EMNIST dataset. The main task is to fine-tune hyperparameters using Genetic Algorithms and Bayesian Optimization to achieve the best performance measured by the macro-averaged F1 score.

### Dataset
The dataset used is a subset of the Federated EMNIST dataset, specifically focusing on digit images (10 classes). It includes:
- train_X.npy
- train_y.npy
- test_X.npy
- test_y.npy

### Neural Network Model
The model is a simple 2-layer neural network with one hidden layer consisting of 128 units.

### Structure
- Input Layer: Size varies depending on the flattened image size.
- Hidden Layer: 128 units.
- Output Layer: 10 units (one for each class).

### Hyperparameter Tuning
Two methods are used for hyperparameter tuning:
1. **Genetic Algorithm**: Custom implementation without using specialized packages.
2. **Bayesian Optimization**: Implemented using the `bayes_opt` package.

### Parameters Tuned
- Mini-batch size (From 16 to 1024)
- Activation function for the hidden layer (ReLU, Sigmoid, Tanh)

Running the Code
1. Install the required packages from `requirements.txt`.
2. You can either run the code for part 1 or 2:
- Part 1: `python 04_hw_q1_code.py`
- Part 2: `python 04_hw_q2_code.py`
3. Results will be displayed in the console and plots will be saved in the `artifacts` folder.

### Optimization Results
**Training Plots:**
Genetic Algorithm
![Genetic Algorithm Training](artifacts/plots/fitness_plot_gen_200.png)
Bayesian Optimization
![Bayesian Optimization Training](artifacts/plots/bo_process.png)

### Testing Results
**Best hyperparameters found by algorithm:**
Genetic Algorithm:
- Batch Size: `21`
- Activation Function: `relu`
Bayesian Optimization:
**Training F1 scores:**
- Batch Size: `25`
- Activation Function: `tanh`
**Comparative analysis of both tuning methods:**
