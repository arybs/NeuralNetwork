import pickle
import numpy as np
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # open data
    with open("./mnist.pkl", "rb") as f:
        data = pickle.load(f)
    # data is dictionary with labels as training_images, training_labels, test_images, test_labels

    train_data = data["training_images"]
    train_data = train_data / 255

    train_labels = data["training_labels"]

    test_labels = data["test_labels"]
    test_data = data["test_images"]
    test_data = test_data / 255

    n = NeuralNetwork(train_data[0].size, 10, [100, 50], 2)
    n.train(train_data, train_labels, epochs=5, do_batches=True)

    counter = 0
    whole = 0
    for i in range(train_data.shape[0]):
        if n.classify(train_data[i]) == train_labels[i]:
            counter += 1
        whole += 1
    print(counter / whole)
    counter = 0
    whole = 0
    for i in range(test_data.shape[0]):
        if n.classify(test_data[i]) == test_labels[i]:
            counter += 1
        whole += 1
    print(counter / whole)
