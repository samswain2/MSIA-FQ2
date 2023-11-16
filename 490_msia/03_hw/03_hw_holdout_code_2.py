import numpy as np
import tensorflow as tf
import os

def load_test_data(test_data_path):
    test_data_array = np.load(test_data_path, allow_pickle=True)
    test_data = test_data_array[0]  # Extract the dictionary from the array
    X_test = np.array(test_data['images'])
    y_test = np.array(test_data['labels'])
    return X_test, y_test

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return loss, accuracy

def main():
    test_data_path = "./Assignment3-data/test_data.npy"
    saved_model_path = './saved_models/global_model_round_80.h5'  # Modify as needed

    X_test, y_test = load_test_data(test_data_path)
    model = load_model(saved_model_path)
    
    loss, accuracy = evaluate_model(model, X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
