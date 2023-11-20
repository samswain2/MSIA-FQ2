import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, test_data_path, parallel=False):
        self.test_data_path = test_data_path
        self.model_dir = './saved_models_parallel' if parallel else './saved_models'
        self.parallel = parallel

    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)

    def load_test_data(self):
        test_data = np.load(self.test_data_path, allow_pickle=True)
        test_images = np.array(test_data[0]['images'])
        test_labels = np.array(test_data[0]['labels'])
        return test_images, test_labels

    def evaluate_model(self, model, test_images, test_labels):
        predictions = model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        correct_predictions = np.sum(predicted_labels == test_labels)
        test_accuracy = correct_predictions / len(test_labels)
        return test_accuracy

    def run_evaluation(self, fraction, output_file, iteration):
        model_path = f"{self.model_dir}/fraction_{fraction}/global_model_round_{iteration}.h5"
        if not tf.io.gfile.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
        model = self.load_model(model_path)

        test_images, test_labels = self.load_test_data()

        test_accuracy = self.evaluate_model(model, test_images, test_labels)
        print(f"Test Accuracy: {test_accuracy}")

        model_type = 'Parallel' if self.parallel else 'Sequential'
        
        with open(output_file, 'a') as file:
            file.write("=============================================\n")
            file.write(f"Evaluation Results for {model_type}-Trained Model - Client Fraction: {fraction}\n")
            file.write("=============================================\n")
            file.write(f"Test Images Shape: {test_images.shape}\n")
            file.write(f"Test Labels Shape: {test_labels.shape}\n")
            file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            file.write("---------------------------------------------\n\n")

        print(f"Evaluation complete for {model_type} model, fraction {fraction}. Results written to {output_file}")
        return test_accuracy


def plot_results(results, plot_file):
    # Sort fractions to ensure the plot is ordered
    fractions = sorted(list(results.keys()))
    accuracies = [results[fraction] for fraction in fractions]  # Access the scalar directly

    # Set the width of each bar to have a small gap between them if desired
    bar_width = 0.01  # Adjust the width to have a tiny gap between bars

    plt.figure(figsize=(10, 5))
    plt.bar(fractions, accuracies, width=bar_width, color='purple', alpha=1)
    
    plt.xlabel('Client Fraction')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy by Client Fraction')
    
    # This ensures that each fraction is labeled and centered on the x-axis
    plt.xticks(fractions, labels=[str(fraction) for fraction in fractions])
    
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(plot_file)

if __name__ == "__main__":
    test_data_path = './Assignment3-data/test_data.npy'

    # Check for command line arguments for sequential or parallel evaluation
    if len(sys.argv) > 1 and sys.argv[1] == '2':
        is_parallel = True
    else:
        is_parallel = False

    # Define the client fractions and file naming based on the value of is_parallel
    client_fractions = [0.04] if is_parallel else [0.025, 0.05, 0.075, 0.1]
    file_tag = 'parallel' if is_parallel else 'sequential'
    output_file = f'./evaluation_results_{file_tag}.txt'
    plot_file = f'./evaluation_plot_{file_tag}.png'

    # Evaluator object instantiation
    evaluator = ModelEvaluator(test_data_path, parallel=is_parallel)

    # Clear the output file
    open(output_file, 'w').close()

    # Evaluate each model
    evaluation_results = {}
    for fraction in client_fractions:
        accuracy = evaluator.run_evaluation(fraction, output_file, iteration=1000)
        if accuracy is not None:
            evaluation_results[fraction] = accuracy

    # Plotting and saving the results
    plot_results(evaluation_results, plot_file)
