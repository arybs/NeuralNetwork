import numpy as np


def logistic_function(x):
    return 1. / (np.exp(-x) + 1)

def logistic_function_derivative(x):
    return logistic_function(x)*(1-logistic_function(x))

