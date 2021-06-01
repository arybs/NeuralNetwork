import pickle
import os
import numpy as np
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
from k_cross_validation import k_cross_validation
import pandas as pd
from neural_network import beta_batch
from neural_network import activation_function_name

columns_names = ["Activation function", "Numbers of epochs", "Beta", "Batch size", "Number of layers", "Layers sizes",
                 "Validation result"]

# params for calculating

numbers_of_layers = 2
layers_list = [40, 40]
beta_to_save = beta_batch
epochs_params = 10
batch_size = 10

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

    if os.path.exists("./Wyniki.csv"):
        df = pd.read_csv("./Wyniki.csv")
    else:
        df = pd.DataFrame(columns=columns_names)

    n = NeuralNetwork(train_data[0].size, 10, layers_list, numbers_of_layers)

    # show validation example
    res = k_cross_validation(train_data, train_labels, n, epochs=epochs_params, batch_size=batch_size)

    to_append = [activation_function_name, epochs_params, beta_to_save, batch_size, numbers_of_layers, layers_list, res]
    t = pd.Series(to_append, index=df.columns)
    df = df.append(t, ignore_index=True)

    df.to_csv("./Wyniki.csv", index=False)

    print("Dopasowanie:")
    print(res)

    '''
    n.train(train_data, train_labels, epochs=5, do_batches=True)

    counter = 0
    whole = 0
    for i in range(train_data.shape[0]):
        if n.classify(train_data[i]) == train_labels[i]:
            counter += 1
        whole += 1
    print("Ile dobrze treningowych")

    print(counter / whole)
    counter = 0
    whole = 0
    for i in range(test_data.shape[0]):
        if n.classify(test_data[i]) == test_labels[i]:
            counter += 1
        whole += 1

    print("Ile testowych robi git")
    print(counter / whole)
    '''
