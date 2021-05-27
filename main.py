import pickle
import numpy as np
from neural_network import NeuralNetwork
if __name__ == "__main__":
    # open data
    with open("./mnist.pkl", "rb") as f:
        data = pickle.load(f)
    # data is dictionary with labels as training_images, training_labels, test_images, test_labels
    # to get some keys:
    print(data.keys())
    # to get train data for example
    train_data = data["training_images"]
    # all data are stored in np.arrays, check NumPy documentation
    print(train_data.shape)  # shape
    print(train_data[0])  # show data


    train_labels = data["training_labels"]

    # just basic things
    n = NeuralNetwork(train_data[0].size, 10, 100)
    n.stochastic_gradient(train_data[:100], train_labels[:100])

    

