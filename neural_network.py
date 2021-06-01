import numpy as np
import activation_functions
from sklearn.utils import shuffle
from typing import List

# define activation_functions
activation_function = activation_functions.logistic_function
activation_function_derivative = activation_functions.logistic_function_derivative

activation_function = activation_functions.ReLU
activation_function_derivative = activation_functions.ReLU_derivative

epochs_number = int(1e3)
beta = 0.01
beta_batch = 0.1

'''
Problem klasyfikacji, oczekujemy na wyjścia wektora samych zer z jedną jedynka (oznaczająca label), do którego klasyfikujemy zdjęcie
Siec jako argument przyjmuje ilosc parametrow na wyjsciu, ilosc kategorii na wyjsciu + liczba ukrytych warstw + lista z iloscia 
neuronow w ukrytych warstwach
'''


# 2 layer version

class NeuralNetwork:
    def __init__(self, numbers_of_params: int, number_of_labels: int, hidden_layer_sizes: List[int],
                 numbers_of_hidden_layers):
        # TODO check if arguments are valid e.g numbers_of_hidden_layers = 3 , hidden_layer_sizes = [3, 2]
        self.first_layer = None
        self.output_layer = None
        self.hidden_layers = [None for _ in range(numbers_of_hidden_layers)]
        self.weights = []
        self.weights.append(
            2 * (np.random.random([hidden_layer_sizes[0] + 1, numbers_of_params + 1]) - 0.5) * 1 / np.sqrt(
                numbers_of_params))
        for iter in range(numbers_of_hidden_layers):
            if iter == numbers_of_hidden_layers - 1:
                self.weights.append(
                    2 * (np.random.random([number_of_labels, hidden_layer_sizes[iter] + 1]) - 0.5) * 1 / np.sqrt(
                        numbers_of_params))
            else:
                self.weights.append(2 * (np.random.random(
                    [hidden_layer_sizes[iter + 1] + 1, hidden_layer_sizes[iter] + 1]) - 0.5) * 1 / np.sqrt(
                    numbers_of_params))

    def count_values_in_layers(self):
        for iter in range(len(self.hidden_layers)):
            if iter == 0:
                self.hidden_layers[iter] = activation_function(np.matmul(self.weights[0], self.first_layer.T))
            else:
                self.hidden_layers[iter] = activation_function(
                    np.matmul(self.weights[iter], self.hidden_layers[iter - 1].T))
            self.hidden_layers[iter][-1] = 1

        self.output_layer = np.matmul(self.weights[-1], self.hidden_layers[-1].T)

    def set_value_in_first_layer(self, values: np.ndarray):
        self.first_layer = values
        self.first_layer = np.append(self.first_layer, 1)

    def count_cost_function(self, expected_result):
        if not isinstance(self.output_layer, np.ndarray):
            self.count_values_in_layers()
        temp = np.zeros(self.output_layer.size)
        temp[expected_result] = 1
        return np.linalg.norm(temp - self.output_layer)

    def gradient(self, expected_result):
        if not isinstance(self.output_layer, np.ndarray):
            self.count_values_in_layers()

        try:
            res_grad = []
            temp = np.zeros(self.output_layer.size)
            temp[expected_result] = 1
            y_temp = 2 * (self.output_layer - temp)
            res_grad.append(np.outer(y_temp, self.hidden_layers[-1]))
            # back propagation
            for iter in range(len(self.hidden_layers) - 1, 0, -1):
                y_temp = np.matmul(y_temp, self.weights[iter + 1])  # vector
                dq_ds = y_temp * activation_function_derivative(
                    np.matmul(self.weights[iter], self.hidden_layers[iter - 1].T))  # vector
                res_grad.append(np.outer(dq_ds, self.hidden_layers[iter - 1]))

            y_temp = np.matmul(y_temp, self.weights[1])  # vector
            dq_ds = y_temp * activation_function_derivative(np.matmul(self.weights[0], self.first_layer.T))  # vector
            res_grad.append(np.outer(dq_ds, self.first_layer))
            res_grad.reverse()
            return res_grad
        except Exception as e:
            print(f"{e.__class__}exception occured")

    def stochastic_gradient(self, train_data: np.ndarray, train_expected: np.ndarray):
        temp = []
        try:
            for iter in range(train_data.shape[0]):
                self.set_value_in_first_layer(train_data[iter])
                self.count_values_in_layers()
                gradient = self.gradient(int(train_expected[iter]))
                for iter1 in range(len(gradient)):
                    self.weights[iter1] -= beta * gradient[iter1]
                temp.append(self.count_cost_function(int(train_expected[iter])))
            return temp
        except Exception as e:
            print(f"{e.__class__}exception occured")
            print(iter)

    def stochastic_gradient_with_mini_batches(self, train_data: np.ndarray, train_expected: np.ndarray,
                                              batches_size: int):
        temp = []
        try:
            iter = 0
            while iter + batches_size < train_data.shape[0]:
                grad_list = []
                temp1 = []
                for iter_batch in range(batches_size):
                    self.set_value_in_first_layer(train_data[iter + iter_batch])
                    self.count_values_in_layers()
                    gradient = self.gradient(int(train_expected[iter + iter_batch]))
                    grad_list.append(gradient)
                    temp1.append(self.count_cost_function(int(train_expected[iter + iter_batch])))

                gradient_avg = np.mean(grad_list, axis=0)

                for iter1 in range(len(gradient_avg)):
                    self.weights[iter1] -= beta_batch * gradient_avg[iter1]
                temp.append(np.mean(temp1))
                iter += batches_size
            return temp
        except Exception as e:
            print(f"{e.__class__}exception occured")

    def train(self, train_data: np.ndarray, train_label: np.ndarray, epochs: int = epochs_number,
              do_batches: bool = False, batch_size: int = 10):
        print("Calculations started")
        for _ in range(epochs):
            to_train_data, to_train_label = shuffle(train_data, train_label)
            
            if do_batches:
                ret = self.stochastic_gradient_with_mini_batches(to_train_data, to_train_label, batch_size)
            else:
                ret = self.stochastic_gradient(to_train_data, to_train_label)
            # at the end next 3 lines could be deleted
            if _ % 10 == 0 or epochs < 100:
                print(_ / epochs * 100)
                print("Cost:\t", np.mean(np.array(ret)))


    def classify(self, data):
        """
        Do only after stochastic_gradient!!!
        :param data: data to classify
        :return: result
        """

        self.set_value_in_first_layer(data)
        self.count_values_in_layers()
        return np.argmin(abs(1 - self.output_layer))
