from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from neural_network import NeuralNetwork
import copy

# Define k parameter
k = 3


def k_cross_validation(data: np.ndarray, expected: np.ndarray, n: NeuralNetwork, epochs: int = 10, batch_size: int = 2):
    kf = KFold(n_splits=k, shuffle=True)
    res = []
    for train_index, test_index in kf.split(data):
        y_predicted = []
        to_train_data, to_train_res = data[train_index], expected[train_index]
        to_test_data, to_test_res = data[test_index], expected[test_index]
        temp = copy.deepcopy(n)
        temp.train(to_train_data, to_train_res, epochs=epochs, do_batches=True, batch_size=batch_size)
        for test_iter in range(len(to_test_data)):
            y_predicted.append(temp.classify(to_test_data[test_iter]))
        res.append(accuracy_score(to_test_res, y_predicted))
    return np.mean(res)
