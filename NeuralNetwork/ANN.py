import numpy as np
from Layer import Layer
from backpropagation import backpropagation

class ANN():

    def __init__(self, layers=None):
        if layers == None:
            self.layers = []
        self.input = None
        self.output = None
        self.target = None
        self.error = None
        self.desired_error = None
    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_momentum(self, momentum):
        self.momentum = momentum

    def set_erf(self, erf):
        self.erf = erf

    def set_desired_error(self, desired_error):
        self.desired_error = desired_error

    def addLayer(self, layer_size, layer_out, activation_function):
        new_layer = Layer(layer_size, layer_out, activation_function)
        new_layer.initialize_weights()
        new_layer.initialize_bias()

        self.layers.append(new_layer)

    def run(self, input):
        self.input = np.array(input)
        layer_forward = self.layers[0].forward_pass(input)
        for i in range(1, len(self.layers) - 1, 1):
            layer_forward_activated = self.layers[i].activate(layer_forward)
            layer_forward = self.layers[i].forward_pass(layer_forward_activated)
        self.output = self.layers[len(self.layers) - 1].activate(layer_forward)

    def return_output(self):
        return self.output

    def calculate_error(self):
        self.target = np.array(self.target)

        if erf == "MSE":
            mse = 0
            for i in range(0, self.layers[len(layers) - 1], 1):
                mse += (self.target[i] - self.output[i])
            mse /= 2
            self.error = mse


    def loss_gradient(self):
        if erf == "MSE":
            return self.target - self.output

    def train(self, input, target):
        for epoch in range(self.max_epoch):
            self.run(input)
            calculate_error()
            if(self.error < self.desired_error):
                break

            self = backpropagation(self, target)
            for i in range(0, len(self.layers) - 1, 1):
                self.layers[i].weight_update = self.layers[i].dWeight + self.layers[i].weight
                self.layers[i].update_weights()
                self.layers[i].update_bias()


    def batch_train(self, data_set, targets, batch_size):
        batch_coeff = 1 / batch_size
        for epoch in range(self.max_epoch):
            for input_data in range(0, len(data_set), 1):
                self.run(data_set[input_data])
                self = backpropagation(self, targets[input_data])
            for i in range(0, len(self_layers) - 1, 1):
                self.layers[i].weight_update = batch_coeff * self.layers[i].delta_weights + self.momentum * self.layers[i].weight
                self.layers[i].update_weights()
                self.layers[i].update_bias()


if __name__ == "__main__":
    test = ANN()
    test.addLayer(3, 4, "input")
    test.addLayer(4, 3, "sigmoid")
    test.addLayer(3, "output", "sigmoid")

    test.run([1,1,1])
    output = test.return_output()
    print(output)
