import numpy as np;
import random
import math

class Layer():
    def __init__(self, prev_layer_size, layer_size, num_layer_out, activation_function):
        self.prev_layer_size = prev_layer_size
        self.layer_size = layer_size
        self.num_layer_out = num_layer_out
        self._activation_function = activation_function

        np.random.seed(0)

    def activation_function(self, x):
        if(self._activation_function == "sigmoid"):
            return 1 / (1 + np.exp(x))
        elif(self._activation_function == "tanh"):
            return np.tanh(x)
        elif(self._activation_function == "linear"):
            return x
        #elif(self.activation_function == "relu"):
        #    return x[x<0] =0

    def random_real(self):
        n = self.layer_size
        upper = math.sqrt(2. / n)
        lower = (-1) * upper
        return random.uniform(lower, upper)

    def initialize_weights(self):
        W = []
        for i in range(0, self.num_layer_out, 1):
            Wi = []
            for j in range(0, self.layer_size, 1):
                Wij = self.random_real()
                Wi.append(Wij)

            W.append(Wi)
        self.weight = np.array(W)

    def initialize_bias(self):
        B = []
        for i in range(0, self.num_layer_out, 1):
            B.append(self.random_real())
        self.bias = np.array(B)

    def forward_pass(self, prev_z):
        self.z_values = self.weight.dot(prev_z) + self.bias
        self.a_values = self.activation_function(self.z_values)
        return self.a_values

    def return_neuron_z_values(self):
        return self.z_values

    def return_neuron_activation_values(self):
        return self.a_values


if __name__ == "__main__":
    layer = Layer(3,3,4, "linear")
    layer.initialize_weights()
    layer.initialize_bias()
    input = np.array([1,0,0])
    output = layer.forward_pass(input)
    print(output)
