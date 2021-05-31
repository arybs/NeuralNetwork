import numpy as np
import activation_functions
from sklearn.utils import shuffle
# define activation_functions
activation_function = activation_functions.logistic_function
activation_function_derivative = activation_functions.logistic_function_derivative

activation_function = activation_functions.ReLU
activation_function_derivative = activation_functions.ReLU_derivative
'''
Notatki:
ogólnie na razie stworzony jest pereptorn dwuwarstowy warto by zrobic go jednak n-warstwowym
Kod jest totalnie nie optymalny, ale za to w miare przejrszyty
Do zrobienia na pewno są jakieś testy + walidacja!!!
Nie wiem na czym skoncze, więc nie wiem czy będzie skończyony perceptron
Ogolnie jakbys dla rade przygotowac sie do walidacji krzyzowej to super bo bedzie trzeba roztrzaskac xD
Jak mozesz to na slajdach w prezce (14, 15 slajd) są wzory związane z gradientami u mnnie. Trzeba to przetestowac czy
nie pomylilem sie jakos w indeksach, mozna na jaims przykladzie
gdzie recznie sie obliczy, bede turbo wdziecnzy za sprawdzenie tego bo ja juz nie wymysle.
Do wymyslenia jak zrobic to tak 

nie sprawdzilem wgl gradientu czy sie kompiluje, on chyba odpowiada za uczenie.
Jak dasz rade przeteuj dla powiedmzy poczaktowych 50 punktow
i zobacz czy cos lepiej ocenia

'''

epochs_number = int(1e6)
# 2 layer version

class NeuralNetwork:
    def __init__(self, numbers_of_params: int, number_of_labels: int, hidden_layer_size: int):
        # TODO make it for n hidden layers
        self.first_layer = None
        self.output_layer = None
        self.hidden_layer = None

        self.weights_first = 2 * (np.random.random([hidden_layer_size + 1, numbers_of_params + 1]) - 0.5) * 1 / np.sqrt(
            numbers_of_params)
        self.weights_second = 2 * (np.random.random([number_of_labels, hidden_layer_size + 1]) - 0.5) * 1 / np.sqrt(
            hidden_layer_size)

    def count_values_in_layers(self):
        self.hidden_layer = activation_function(np.matmul(self.weights_first, self.first_layer.T))
        self.hidden_layer[-1] = 1
        self.output_layer = np.matmul(self.weights_second, self.hidden_layer.T)

    def set_value_in_first_layer(self, values: np.ndarray):
        self.first_layer = values
        self.first_layer = np.append(self.first_layer, 1)

    def count_cost_function(self, expected_result):
        if not isinstance(self.output_layer, np.ndarray):
            self.count_values_in_layers()
        temp = np.zeros(self.output_layer.size)
        temp[expected_result] = 1
        return np.linalg.norm(temp - self.output_layer)

    def gradient_output_layer(self, expected_result):
        if not isinstance(self.output_layer, np.ndarray):
            self.count_values_in_layers()

        try:
            temp = np.zeros(self.output_layer.size)
            temp[expected_result] = 1
            return 2 * np.outer((self.output_layer - temp), self.hidden_layer)
        except Exception as e:
            print(f"{e.__class__}exception occured")

    def gradient_hidden_layers(self, expected_result):
        temp = np.zeros(self.output_layer.size)
        temp[expected_result] = 1
        dq_dy1 = np.matmul((2 * (self.output_layer - temp)), self.weights_second)  # vector
        dq_ds = dq_dy1 * activation_function_derivative(np.matmul(self.weights_first, self.first_layer.T))  # vector
        return np.outer(dq_ds, self.first_layer)

    def stochastic_gradient(self, train_data: np.ndarray, train_expected: np.ndarray):
        # TODO podobno to sie oplaca robic jako losowanie ze zwracaniem tych probek, ale kiedy to konczyc? Chyba dziala tak, ze my 'udajemy ze to losujemy'
        # ale pewnosci nie mam.
        beta = 0.01  # staly krok można zmienić
        temp = []
        try:
            for iter in range(train_data.shape[0]):
                self.set_value_in_first_layer(train_data[iter])
                self.count_values_in_layers()
                grad_output = self.gradient_output_layer(int(train_expected[iter]))
                # self.output_layer = self.output_layer/(np.max(self.output_layer)-np.min(self.output_layer))
                grad_hidden = self.gradient_hidden_layers(int(train_expected[iter]))
                self.weights_second -= beta * grad_output
                self.weights_first -= beta * grad_hidden
                temp.append(self.count_cost_function(int(train_expected[iter])))
            return temp
        except Exception as e:
            print(f"{e.__class__}exception occured")
            print(iter)

    def train(self, train_data: np.ndarray, train_label: np.ndarray, epochs: int = epochs_number):
        for _ in range(epochs):
            to_train_data, to_train_label = shuffle(train_data, train_label)
            temp = self.stochastic_gradient(to_train_data, to_train_label)
            print(_/epochs*100, f"\tCost {np.mean(temp)}")


    def classify(self, data):
        '''
        Do only after stochastic_gradient!!!
        :param data: data to classify
        :return: result
        '''
        self.set_value_in_first_layer(data)
        self.count_values_in_layers()
        return np.argmin(abs(1-self.output_layer))


'''

# let's test

nn = NeuralNetwork(4, 4, 13)
a = np.array([1, 2, 3, 4])
nn.set_value_in_first_layer(a)
print(nn.first_layer)
nn.count_values_in_layers()
print(nn.output_layer)

print(nn.count_cost_function(2))

#print(nn.weights_second - nn.gradient_output_layer(2))

print(nn.gradient_hidden_layers(1) )
'''
