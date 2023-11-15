import tensorflow as tf
import numpy as np

class ModelEvaluator:
    def __init__(self, model_path, test_data_path, output_file):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.output_file = output_file

    def load_model(self):
        return tf.keras.models.load_model(self.model_path)

    def load_test_data(self):
        test_data = np.load(self.test_data_path, allow_pickle=True)
        test_images = np.array(test_data[0]['images'])
        test_labels = np.array(test_data[0]['labels'])
        return test_images, test_labels

    def evaluate_model(self, model, test_images, test_labels):
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
        return test_loss, test_accuracy

    def write_results_to_file(self, test_loss, test_accuracy):
        with open(self.output_file, 'w') as file:
            file.write(f"Test Loss: {test_loss}\n")
            file.write(f"Test Accuracy: {test_accuracy}\n")

    def run_evaluation(self):
        model = self.load_model()
        test_images, test_labels = self.load_test_data()
        test_loss, test_accuracy = self.evaluate_model(model, test_images, test_labels)
        self.write_results_to_file(test_loss, test_accuracy)
        print(f"Evaluation complete. Results written to {self.output_file}")

# Configuration
parallel = 'parallel'
model_path = './saved_models/global_model_round_200.h5'  # Adjust as needed
test_data_path = './Assignment3-data/test_data.npy'    # Adjust as needed
output_file = './evaluation_results_regular.txt'               # Output file path

# Create an instance of the evaluator and run the evaluation
evaluator = ModelEvaluator(model_path, test_data_path, output_file)
evaluator.run_evaluation()
