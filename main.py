import pickle
import numpy as np
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
#from sklearn import preprocessing

def normalize_data (data: np.ndarray):
    num_parameters = data[0].size
    expected_values = np.zeros(num_parameters)
    variance = np.zeros(num_parameters)
    number = int(data.size/float(num_parameters))
    for i in range(number):
        for j in range(num_parameters):
            expected_values[j] += data[i][j]/float(number)
    
    for i in range(number):
        for j in range(num_parameters):
            variance[j] += ((data[i][j] - expected_values[j])**2)/float(number)
    
    for i in range(number):
        for j in range(num_parameters):
            if (variance[j] != 0):
                data[i][j] = (data[i][j] - expected_values[j])/(variance[j]**(0.5))
            else:
                data[i][j] = 0 




if __name__ == "__main__":
    # open data
    with open("./mnist.pkl", "rb") as f:
        data = pickle.load(f)
    # data is dictionary with labels as training_images, training_labels, test_images, test_labels
    # to get some keys:
    print(data.keys())
    # to get train data for example
    train_data = data["training_images"]
    train_data = train_data/ 255
    # all data are stored in np.arrays, check NumPy documentation
    print(train_data.shape)  # shape
    print(train_data[0])  # show data

    print(np.max(train_data))

    train_labels = data["training_labels"]

    # just basic things
    # normalization goes like this x =  (x-mean(x))/std(x)
    #normalize_data(train_data[:100])
    #print(train_data[0])

    n = NeuralNetwork(train_data[0].size, 10, 200)
    #ret = n.stochastic_gradient(train_data[:10000], train_labels[:10000])
    #x = np.linspace(0, 9999, 10000)
    #x = x[::1000]
    #y = ret[::1000]
    #print(y)
    #plt.plot(np.linspace(0, 999, 1000)[:100], ret[:100])
    #plt.plot(x, y)
    #plt.show()
    n.train(train_data[:10000], train_labels[:10000], epochs=25)
    counter = 0
    whole = 0
    for i in range(train_data[:10000].shape[0]):
        if n.classify(train_data[i]) == train_labels[i]:
            counter += 1
        whole += 1
    print(counter/whole)

