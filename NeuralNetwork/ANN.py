import numpy as np
from Layer import Layer
import backpropagation

class ANN():

    def __init__(self, layersinit=None):
        if layersinit == None:
            self.layers = []
        self._input = None
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

    def addLayer(self, layer_size, num_layer_out, activation_function):
        new_layer = Layer(layer_size, num_layer_out, activation_function)
        new_layer.initialize_weights()
        new_layer.initialize_bias()

        self.layers.append(new_layer)

    def run(self, _input):
        if self._input == None:
            self._input = np.transpose(np.array([_input]))

        layer_forward = self.layers[0].forward_pass(self._input)
        for i in range(1, len(self.layers) - 1, 1):
            layer_forward = self.layers[i].forward_pass(layer_forward)


        self.output = self.layers[len(self.layers) - 1].activate(layer_forward)



    def calculate_error(self):

        if self.erf == "MSE":
            mse = 0
            for i in range(0, len(self.output[0]) - 1, 1):
                mse += (self.target[0][i] - self.output[0][i])
            mse /= 2
            self.error = mse

    def loss_gradient(self):
        if self.erf == "MSE":
            return self.target - self.output

    def train(self, _input, target):
        if self.target == None:
            self.target = np.transpose(np.array([target]))

        for epoch in range(0, self.max_epoch, 1):
            self.run(_input)
            print("====OUTPUT====")
            print(self.output)
            print("---loss gradient---")
            print(self.loss_gradient())
            print("==============")
            self.calculate_error()
            #if(self.error < self.desired_error):
            #    break
            self = backpropagation.backpropagation(self)
            for i in range(0, len(self.layers) - 1, 1):
                self.layers[i].weight_update = self.layers[i].dWeight + self.layers[i].weight
                #print(self.layers[i].weight_update)
                self.layers[i].update_weights(self.learning_rate)
                self.layers[i].update_bias(self.learning_rate)
            self._input = None
            self.output = None
            for i in range(0, len(self.layers) - 1, 1):
                self.layers[i].z_values = None
                self.layers[i].a_values = None
                self.layers[i].delta = None
        self.target = None
