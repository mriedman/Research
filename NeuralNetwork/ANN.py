import numpy as np
from Layer import Layer
import backprop

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
            for i in range(0, len(self.output[0]), 1):
                mse += (self.target[0][i] - self.output[0][i])**2
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
            self.calculate_error()
            print("====EPOCH " + str(epoch + 1) + " ====")
            print("---OUTPUT---")
            print(self.output)
            print("---ERROR---")
            print(self.error)
            print("==============")

            if(self.error < self.desired_error):
                break
            self = backprop.backpropagation(self)
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

    def batch_train(self, data_set, targets, batch_size):
        batch_coeff = 1 / batch_size
        for epoch in range(self.max_epoch):
            for input_data in range(0, len(data_set), 1):
                if self.target == None:
                    self.target = np.transpose(np.array([targets[input_data]]))
                self.run(data_set[input_data])
                self.calculate_error()
                print("====EPOCH " + str(epoch + 1) + "====")
                print("++input " + str(input_data + 1) + "++")
                print("---OUTPUT---")
                print(self.output)
                print("---ERROR---")
                print(self.error)
                print("==============")

                self = backprop.backpropagation(self)

                self._input = None
                self.output = None
                self.target = None
                for i in range(0, len(self.layers) - 1, 1):
                    self.layers[i].z_values = None
                    self.layers[i].a_values = None
                    self.layers[i].delta = None

            for i in range(0, len(self.layers) - 1, 1):
                self.layers[i].weight_update = batch_coeff * self.layers[i].dWeight + self.momentum * self.layers[i].weight
                self.layers[i].dBias = batch_coeff
                self.layers[i].update_weights(self.learning_rate)
                self.layers[i].update_bias(self.learning_rate)
