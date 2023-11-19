import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, test_data_path):
        self.test_data_path = test_data_path

    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)

    def load_test_data(self):
        test_data = np.load(self.test_data_path, allow_pickle=True)
        test_images = np.array(test_data[0]['images'])
        test_labels = np.array(test_data[0]['labels'])
        return test_images, test_labels

    def evaluate_model(self, model, test_images, test_labels):
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
        return test_loss, test_accuracy

    def run_evaluation(self, model_path, output_file):
        model = self.load_model(model_path)
        test_images, test_labels = self.load_test_data()
        test_loss, test_accuracy = self.evaluate_model(model, test_images, test_labels)

        with open(output_file, 'a') as file:
            file.write(f"Model: {model_path}\n")
            file.write(f"Test Loss: {test_loss}\n")
            file.write(f"Test Accuracy: {test_accuracy}\n\n")

        print(f"Evaluation complete for {model_path}. Results written to {output_file}")
        return test_loss, test_accuracy

def plot_results(results, plot_file):
    fractions = list(results.keys())
    accuracies = [results[fraction]['accuracy'] for fraction in fractions]
    losses = [results[fraction]['loss'] for fraction in fractions]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Client Fraction')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.bar(fractions, accuracies, color='tab:blue', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')
    ax2.plot(fractions, losses, color='tab:red', marker='o')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.savefig(plot_file)  # Save the plot to a file
    plt.show()


# Configuration
test_data_path = './Assignment3-data/test_data.npy'  # Adjust as needed
output_file = './evaluation_results.txt'  # Output file path

# Client fractions and corresponding model paths
client_fractions = [0.025, 0.05, 0.075, 0.1]
model_paths = {fraction: f'./saved_models/fraction_{fraction}/global_model_round_125.h5' for fraction in client_fractions}

# Clearing the output file
open(output_file, 'w').close()

# Evaluate each model
evaluator = ModelEvaluator(test_data_path)
evaluation_results = {}

for fraction, model_path in model_paths.items():
    test_loss, test_accuracy = evaluator.run_evaluation(model_path, output_file)
    evaluation_results[fraction] = {'loss': test_loss, 'accuracy': test_accuracy}

# Plotting and saving the results
plot_file = './evaluation_plot.png'  # Path to save the plot
plot_results(evaluation_results, plot_file)
