import numpy as np
import activation_functions

# define activation_functions
activation_function = activation_functions.logistic_function
activation_function_derivative = activation_functions.logistic_function_derivative

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


# 2 layer version

class NeuralNetwork:
    def __init__(self, numbers_of_params: int, number_of_labels: int, hidden_layer_size: int):
        # do przemyslenia uogolnienie + czy potrzebna taka inicjalizacja
        self.first_layer = None
        self.output_layer = None
        self.hidden_layer = None
        # TODO: Check how weights should be generated at first
        self.weights_first = np.random.random([hidden_layer_size + 1, numbers_of_params + 1]) - 0.5
        self.weights_second = np.random.random([number_of_labels, hidden_layer_size + 1]) - 0.5

    def count_values_in_layers(self):
        self.hidden_layer = activation_function(np.matmul(self.weights_first, self.first_layer.T))
        self.output_layer = np.matmul(self.weights_second, self.hidden_layer.T)
        self.hidden_layer[-1] = 1

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
        dq_dy1 = np.matmul((2 * (self.output_layer - expected_result)), self.weights_second)  # vector
        dq_ds = dq_dy1 * activation_function_derivative(np.matmul(self.weights_first, self.first_layer.T))  # vector
        return np.outer(dq_ds, self.first_layer)
    def stochastic_gradient(self, train_data: np.ndarray, train_expected: np.ndarray):
        #TODO podobno to sie oplaca robic jako losowanie ze zwracaniem tych probek, ale kiedy to konczyc? Chyba dziala tak, ze my 'udajemy ze to losujemy'
        # ale pewnosci nie mam.
        beta = 0.2 # staly krok można zmienić
        for iter in range(train_data.shape[0]):
            self.set_value_in_first_layer(train_data[iter])
            self.count_values_in_layers()
            self.weights_second -= beta*self.gradient_output_layer(int(train_expected[iter]))
            # pytanie: Jeśli dobrze rozumiem w metodzie wstecznej propagacji jest tak ze najpierw oblcizmy gradient
            # warstw ostatnich i potem gradienty ukrtych zalezy od tych wag dalszych, tylko przed czy po aktualizacji
            self.weights_first -= beta*self.gradient_hidden_layers(int(train_expected[iter]))
    def classify(self, data):
        '''
        Do only after stochastic_gradient!!!
        :param data: data to classify
        :return: result
        '''
        self.first_layer(data)
        self.count_values_in_layers()
        return np.argmax(self.output_layer)

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
