from neural_network import *
nn = NeuralNetwork(3, 2, [4], 1)

a = np.array([3, 1, -4])
b = np.array([0, 0])

nn.set_value_in_first_layer(a)
print("First layer\n")
print(nn.first_layer)
print("First weights\n")
print(nn.weights[0])
nn.count_values_in_layers()
print("hidden layer\n")
print(nn.hidden_layers[0])
print("\nsecond weights\n")
print(nn.weights[1])
print("\noutput layer\n")
print(nn.output_layer)
# grad_output = nn.gradient_output_layer(1)
grad = nn.gradient(1)
grad_output = grad[1]
print("\ngrad_output\n")
print(grad_output)
# grad_hidden = nn.gradient_hidden_layers(1)
grad_hidden = grad[0]
print("\ngrad_hidden\n")
print(grad_hidden)
temp = np.zeros(nn.output_layer.size)
temp[1] = 1
ret = np.linalg.norm(temp - nn.output_layer)
print(ret)

c1 = []
c3 = []
text = []
ex = 0
for i in range(2):
    for j in range(4):
        if i == 1:
            ex = 1
        else:
            ex = 0
        c1.append(2 * nn.output_layer[i] * nn.hidden_layers[0][j])
        if 2 * (nn.output_layer[i] - ex) * nn.hidden_layers[0][j] == grad_output[i][j]:
            text.append("ok")
        else:
            text.append("error")

dqdy1 = 2 * nn.output_layer[0] * nn.weights[1][0, 0] + 2 * (nn.output_layer[1] - 1) * nn.weights[1][1, 0]
dqds1 = dqdy1 * activation_function_derivative(
    nn.weights[0][0, 0] * nn.first_layer[0] + nn.weights[0][0, 1] * nn.first_layer[1] + nn.weights[0][0, 2] *
    nn.first_layer[2] + nn.weights[0][0, 3] * nn.first_layer[3])

c3 = dqds1 * nn.first_layer[0]

print(c1)

print(c3)
print(grad_hidden[0][0])
print(text)
