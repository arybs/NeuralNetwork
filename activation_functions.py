import numpy as np


def logistic_function(x):
    return 1. / (np.exp(-x) + 1)


def logistic_function_derivative(x):
    return logistic_function(x) * (1 - logistic_function(x))


def ReLU(x):
    return np.maximum(0, x)


def ReLU_derivative(x):
    if isinstance(x, np.float):
        if x > 0:
            return 1
        else:
            return 0
    return np.array([1 if elem > 0 else 0 for elem in x])
