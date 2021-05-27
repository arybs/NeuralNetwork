import pickle
import numpy as np
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn import preprocessing

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

    print(np.max(train_data))

    train_labels = data["training_labels"]

    # just basic things
    # normalization goes like this x =  (x-mean(x))/std(x)
    train_data = train_data/255
    print(train_data[0])
    n = NeuralNetwork(train_data[0].size, 10, 100)
    ret = n.stochastic_gradient(train_data[:10000], train_labels[:10000])
    x = np.linspace(0, 9999, 10000)
    x = x[::1000]
    y = ret[::1000]
    #plt.plot(np.linspace(0, 999, 1000)[:100], ret[:100])
    plt.plot(x, y)
    plt.show()
