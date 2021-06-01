from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from neural_network import NeuralNetwork

# Define k parameter
k = 3


def k_cross_validation(data: np.ndarray, expected: np.ndarray, n: NeuralNetwork):
    # to_train_data, to_train_label = shuffle(data, expected)
    kf = KFold(n_splits=k, shuffle=True)
    res = []
    for train_index, test_index in kf.split(data):
        y_predicted = []
        to_train_data, to_train_res = data[train_index], expected[train_index]
        to_test_data, to_test_res = data[test_index], expected[test_index]
        # Available to edit - changing the batches_size or epochs number - just to get the best parameters
        # Maybe add it to a function? Like k-cross_validation could take batch_size, or epochs as an argument!
        n.train(to_train_data, to_train_res, epochs=10, do_batches=True, batch_size=2)
        for test_iter in range(len(to_test_data)):
            y_predicted.append(n.classify(to_test_data[iter]))
        res.append(accuracy_score(to_test_res, y_predicted))
    return np.mean(res)
