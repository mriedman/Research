import numpy as np;

class Layer():
    def __init__(self, prev_layer_size, num_layer_out, activation_function):
        self.prev_layer_size = prev_layer_size
        self.layer_size = layer_size
        self.num_layer_out = num_layer_out
        self.activation_function = activation_function

        np.random.seed(0)

    def activation_function(self, x):
        if(self.activation_function == "sigmoid"):
            return np.sigmoid(x)
        elif(self.activation_function == "tanh"):
            return np.tanh(x)
        elif(self.activation_function == "linear"):
            return x
        elif(self.activation_function == "relu"):
            return x[x<0] =0

    def he_et_al_random():
        return np.random.randn(self.layer_size, self.prev_layer_size)*np.sqrt(2/self.prev_layer_size)

    def initialize_weights(self):
        W = []
        for i in range(0, self.num_layer_out, 1):
            Wi = []
            for j in range(0, self.layer_size, 1):
                Wij = he_et_al_random()
                Wi.append(Wij)
            W.append(Wi)
        self.weight = np.array(W)

    def initialize_bias(self):
        B = []
        for i in range(0, self.layer_size, 1):
            B.append(he_et_al_random)
        self.bias = np.array(B)

    def forward_pass(self):
        self.z_values = self.weight.dot(self.in)
        
    def return_neuron_z_values(self):
        return self.z_values

    def return_neuron_activation_values(self):
        return self.activation_values
