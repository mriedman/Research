import numpy as np
from Layer import Layer
from backpropagation import *

class ANN():

    def __init__(self, layersinit=None):
        if layersinit == None:
            self.layers = []
        self.input = None
        self.output = None
        self.target = None
        self.error = None
        self.desired_error = None

        self.input_shape = None
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

    def addLayer(self, layer_size, num_layer_out, activation_function):
        new_layer = Layer(layer_size, num_layer_out, activation_function)
        new_layer.initialize_weights()
        new_layer.initialize_bias()

        self.layers.append(new_layer)

    def run(self, input):
        self.input = np.array([input])
        layer_forward = self.layers[0].forward_pass(self.input)
        for i in range(1, 2, 1):
            print(i)
            layer_forward_activated = self.layers[i].activate(layer_forward)
            layer_forward = self.layers[i].forward_pass(layer_forward_activated)
        #print(self.layers[1].weight)
        #print(self.layers[1].a_values)
        #print(layer_forward)

        self.output = self.layers[len(self.layers) - 1].activate(layer_forward)

    def get_output(self):
        return self.output

    def calculate_error(self):
        self.target = np.array(self.target)

        if self.erf == "MSE":
            mse = 0
            #print("out: ")
            #print(self.output)
            for i in range(0, len(self.output[0]) - 1, 1):
                mse += (self.target[0][i] - self.output[0][i])
            mse /= 2
            self.error = mse


    def loss_gradient(self):
        if self.erf == "MSE":
            return self.target - self.output

    def train(self, input, target):
        #self.input = np.array(input)
        #self.input_shape = self.input.shape
        #print("shape")
        #print(self.input_shape)
        self.target = np.array([target])
        for epoch in range(0, self.max_epoch, 1):
            print("hello")
            self.run(input)
            print("ran")
            self.calculate_error()
            #if(self.error < self.desired_error):
            #    break
            print("hello")
            self = backpropagation(self)
            print("backprop")
            for i in range(0, len(self.layers) - 1, 1):
                self.layers[i].weight_update = self.layers[i].dWeight + self.layers[i].weight
                #print(self.layers[i].weight_update)
                self.layers[i].update_weights(self.learning_rate)
                self.layers[i].update_bias(self.learning_rate)
            print("update")

    def batch_train(self, data_set, targets, batch_size):
        batch_coeff = 1 / batch_size
        for epoch in range(self.max_epoch):
            for input_data in range(0, len(data_set), 1):
                self.input = np.array([data_set[input_data]])
                self.target = np.array([targets[input_data]])
                self.run(self.input)
                self = backpropagation(self)
            for i in range(0, len(self.layers) - 1, 1):
                self.layers[i].weight_update = batch_coeff * self.layers[i].delta_weights + self.momentum * self.layers[i].weight
                self.layers[i].update_weights()
                self.layers[i].update_bias()


if __name__ == "__main__":
    test = ANN()
    test.addLayer(2, 3, "input")
    test.addLayer(3, 1, "sigmoid")
    test.addLayer(1, "output", "sigmoid")

    test.set_max_epoch(40000)
    test.set_desired_error(0.001)
    test.set_momentum(0.05)
    test.set_learning_rate(0.05)
    test.set_erf("MSE")


    test.train([1, 1], [0])
    input = [1, 1]
    #test.run(input)
    #output = test.get_output()
    #print(output)
