import numpy as np
import random
import math

class Layer():

    def __init__(self, layer_size, num_layer_out, layer_type):
        self.layer_size = layer_size
        self.num_layer_out = num_layer_out
        self.layer_type = layer_type

        self.z_values = None
        self.a_values = None

        self.delta = None
        self.dWeight = None
        self.weight = None
        self.weight_update = None

        self.dBias = None
        np.random.seed(0)

    def activation_function(self, x):
        if self.layer_type == "sigmoid":
            return 1 / (1 + np.exp(x))
        elif self.layer_type == "tanh":
            return np.tanh(x)
        elif self.layer_type == "linear":
            return x
        else:
            return None

    def activate(self, z_vals):
        if self.num_layer_out == "output":
            self.z_values = z_vals
        return self.activation_function(z_vals)


    def activation_function_derivative(self, x):
            if self.layer_type == "sigmoid":
                return self.activation_function(x) * (1 - self.activation_function(x))
            elif self.layer_type == "tanh":
                return (1 - np.tanh(x)**2)
            elif self.layer_type == "linear":
                return 1
            else:
                return None

    def random_real(self):
        n = self.layer_size
        upper = math.sqrt(2. / n)
        lower = (-1) * upper
        return random.uniform(lower, upper)


    def initialize_weights(self):
        if self.num_layer_out != "output":
            #weights
            W = []
            for i in range(0, self.num_layer_out, 1):
                Wi = []
                for j in range(0, self.layer_size, 1):
                    Wij = self.random_real()
                    Wi.append(Wij)

                W.append(Wi)
            self.weight = np.array(W)
            #delta weights for backprop // zero vector
            self.dWeight = np.zeros_like(self.weight)
            self.weight_update = np.zeros_like(self.weight)

        else:
            pass

    def initialize_bias(self):
        if self.num_layer_out != "output":
            B = []
            for i in range(0, self.num_layer_out, 1):
                B.append(self.random_real())
            self.bias = np.transpose(np.array([B]))
            self.dBias = np.zeros_like(self.bias)

        else:
            pass


    def forward_pass(self, layer_in):
        #layer_in IS A COLUMN VECTOR
        #bias IS A COLUMN VECTOR
        self.z_values = layer_in
        if self.layer_type != "input":
            self.a_values = self.activate(self.z_values)
        z_next = np.matmul(self.weight, layer_in) + self.bias
        return z_next

    def set_delta(self, delta):
        self.delta = delta

    def update_weights(self, learning_rate):
        self.weight = self.weight - learning_rate * self.weight_update

    def update_bias(self, learning_rate):

        self.bias = self.bias - learning_rate * self.dBias

    def get_z_values(self):
        return self.z_values

    def get_activation_values(self):
        return self.a_values
