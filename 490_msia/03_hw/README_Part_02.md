# Assignment 3 - Federated Learning (Part 2)

## Differential Privacy Implementation

### Overview
This part of the assignment implements differential privacy for a federated learning task. To ensure the privacy of local data for each client, random noise drawn from a Laplace distribution was injected into the dataset.

### Method
For each 28x28 image $X$ from the Federated EMNIST dataset, noise with scale $b$ was added to each pixel to obtain a perturbed image $X'$, where $X' = X + ε$ and each entry $ε_{ij}$ is sampled from a zero-centered Laplace distribution with scale $b$.

### Experimentation
The effect of noise scale $b$ on the model quality trained via FedAvg was explored with the following steps:
1. Local training on the perturbed dataset for each client.
2. Training a 2-layer Neural Network with 128 hidden units and the RELU activation function.
3. Utilizing 10% of clients in each communication round for FedAvg.
4. Plotting aggregated training and validation accuracy across communication rounds.
5. Evaluating the trained model on held-out test data.
6. Repeat for multiple noise values.

### Code Instructions

To run the code related to this assignment, please cd to the main directory of this assignment and use the cmd command below:

##### Training Script
```cmd
python310 homework_03_part_02/03_hw_02_part_code.py
```

##### Testing Script
```cmd
python310 homework_03_part_02/03_hw_02_part_holdout_code.py
```

- Note: Before running each script, there are some settings you can adjust at the bottom. Please take a look at them before running. 

### Results

#### Training Quality with Different Noise Scales
A plot comparing the training quality with different noise scales is provided. The y-axis represents accuracy, while the x-axis represents the noise scale $b$.

#### Noise Scale Selection
After careful analysis, a noise scale of **0.5** was found to best balance privacy and model quality. This scale ensures adequate privacy while preserving the integrity and performance of the trained model.

### Plots and Evaluation Results
The training progression and final model accuracy plots can be found below:

**Training Plots for Different Noise Levels**
![Training Progression with Noise Scale 0.1](./03_hw_02_part_plots/training_progress_round_500_noise_0.1.png)
![Training Progression with Noise Scale 0.2](./03_hw_02_part_plots/training_progress_round_500_noise_0.2.png)
![Training Progression with Noise Scale 0.5](./03_hw_02_part_plots/training_progress_round_500_noise_0.5.png)
![Training Progression with Noise Scale 1.0](./03_hw_02_part_plots/training_progress_round_500_noise_1.0.png)

**Evaluation Plots for Train and Test Sets**
![Model Accuracy by Noise Scale](./03_hw_02_part_plots/evaluation_plot_sequential_training_data.png)
![Model Accuracy by Noise Scale](./03_hw_02_part_plots/evaluation_plot_sequential_testing_data.png)

The evaluation results are detailed in the attached text file:
- [Evaluation Results Text File](./03_hw_02_part_plots/03_hw_02_part_evaluation_results.txt)

## Analysis of Noise Scale Impact on Model Quality

In the exploration of differential privacy within federated learning, the injection of noise into local training data has shown a clear trade-off between privacy and accuracy. The implementation of FedAvg to train a 2-layer neural network with varied noise scales demonstrated that as the noise scale \( b \) increased, the model's accuracy, both on training and testing datasets, tended to decrease. This trend is indicative of the expected degradation in model performance due to the introduction of noise. Notably, at lower noise scales \( b \leq 0.1 \), the model preserved a high degree of accuracy, suggesting that a minimal noise addition could be a viable strategy for privacy preservation without significant loss in model utility. Conversely, at higher noise scales \( b \geq 0.5 \), the accuracy dropped more steeply, reflecting the model's inability to learn effectively from heavily perturbed data. The accuracy plots across communication rounds indicated an earlier plateau for models trained with higher noise scales, possibly signifying early convergence due to noise-induced underfitting. Considering these findings, a noise scale at the lower end of the spectrum \( b \approx 0.3 \) to \( 0.5 \) might offer an optimal balance between data privacy and model accuracy.

