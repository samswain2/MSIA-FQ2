import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, test_data_path, train_data_path, parallel=False):
        self.test_data_path = test_data_path
        self.train_data_path = train_data_path
        self.model_dir = './saved_models_parallel' if parallel else './saved_models'
        self.parallel = parallel

    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)

    def load_training_data(self, train_data_path):
        train_data = np.load(train_data_path, allow_pickle=True)
        train_images = []
        train_labels = []
        for client_data in train_data:
            train_images.append(client_data['images'])
            train_labels.append(client_data['labels'])
        return np.concatenate(train_images), np.concatenate(train_labels)

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

    def run_evaluation(self, fraction, local_epochs, noise_scale, output_file, iteration, data_type='testing'):
        model_path = f"{self.model_dir}/fraction_{fraction}_epochs_{local_epochs}_noise_{noise_scale}/global_model_round_{iteration}.h5"
        if not tf.io.gfile.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
        model = self.load_model(model_path)

        if data_type == 'testing':
            images, labels = self.load_test_data()
        elif data_type == 'training':
            images, labels = self.load_training_data(self.train_data_path)
        else:
            raise ValueError("Invalid data_type. Choose 'testing' or 'training'.")

        test_accuracy = self.evaluate_model(model, images, labels)
        print(f"Test Accuracy: {test_accuracy}")

        model_type = 'Parallel' if self.parallel else 'Sequential'
        
        with open(output_file, 'a') as file:
            file.write("=============================================\n")
            file.write(f"Evaluation Results for {model_type}-Trained Model - Noise Scale: {noise_scale}\n")
            file.write(f"Data Type: {'Training' if data_type == 'training' else 'Testing'}\n")  # New line for data type
            file.write("=============================================\n")
            file.write(f"Images Shape: {images.shape}\n")
            file.write(f"Labels Shape: {labels.shape}\n")
            file.write(f"Accuracy: {test_accuracy:.4f}\n")
            file.write("---------------------------------------------\n\n")

        print(f"Evaluation complete for {model_type} model with noise scale {noise_scale}. Results written to {output_file}")
        return test_accuracy

def plot_results(results, plot_file, data_type='testing'):
    # Ensure noise_scales is a flat list of unique noise scale values
    noise_scales = sorted(set([key[2] for key in results.keys()]))  # Assuming key is a tuple (fraction, local_epochs, noise_scale)
    # Ensure accuracies is a list with a matching length to noise_scales
    accuracies = [results[(0.1, local_epochs, noise)] for noise in noise_scales for local_epochs in [20]]

    plt.figure(figsize=(10, 5))
    # Plot bars with a loop to handle individual labels and colors
    for i, (noise_scale, accuracy) in enumerate(zip(noise_scales, accuracies)):
        plt.bar(i, accuracy, color='skyblue', label=f'Noise: {noise_scale}')
    
    # Set the x-axis to have ticks at the positions of the bars with the noise scale labels
    plt.xticks(range(len(noise_scales)), labels=[f'{noise}' for noise in noise_scales])
    
    plt.xlabel('Noise Scale')
    plt.ylabel('Accuracy')
    plt.title(f'Model Accuracy by Noise Scale for Fraction 0.1 ({data_type.capitalize()} Data)')
    plt.tight_layout()
    plt.savefig(plot_file)

if __name__ == "__main__":
    test_data_path = './Assignment3-data/test_data.npy'
    train_data_path = './Assignment3-data/train_data.npy'
    data_type = 'testing'

    # Check for command line arguments for sequential or parallel evaluation
    if len(sys.argv) > 1 and sys.argv[1] == '2':
        is_parallel = True
    else:
        is_parallel = False

    # Define the local epoch values to evaluate
    local_epoch_values = [20]

    # Define noise scales to evaluate
    noise_scales = [0.1, 0.2, 0.5, 1.0]

    # Define the client fractions and file naming based on the value of is_parallel
    client_fractions = [0.04] if is_parallel else [0.1] # 0.025, 0.05, 0.075, 
    file_tag = 'parallel' if is_parallel else 'sequential'
    output_file = f'./evaluation_results_{file_tag}.txt'
    plot_file = f'./evaluation_plot_{file_tag}.png'

    # Evaluator object instantiation
    evaluator = ModelEvaluator(test_data_path, train_data_path, parallel=is_parallel)

    # Clear the output file
    open(output_file, 'w').close()

    # Evaluate each model for each combination of fraction, local epochs, and noise scale
    evaluation_results = {}
    for fraction in client_fractions:
        for local_epochs in local_epoch_values:
            for b in noise_scales:
                accuracy = evaluator.run_evaluation(fraction, local_epochs, b, output_file, iteration=500, data_type=data_type)
                if accuracy is not None:
                    evaluation_results[(fraction, local_epochs, b)] = accuracy

    # Plotting and saving the results (if you want to adjust this for different epochs, let me know)
    plot_results(evaluation_results, plot_file, data_type=data_type)
