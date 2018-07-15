import numpy as np
from Layer import Layer

class ANN():

    def __init__(self, layers=None):
        if layers == None:
            self.layers = []
        self.input = None
        self.output = None
        self.target = None
        self.error = None

    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_momentum(self, momentum):
        self.momentum = momentum

    def set_erf(self, erf):
        self.erf = erf

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
            return mse;


    def loss_gradient(self):
        if erf == "MSE":
            return self.target - self.output

    #def train(self, input, output):

    #def batch_train(self, inputs, outputs, batchsize):
    #def train(self, input, output):


if __name__ == "__main__":
    test = ANN()
    test.addLayer(3, 4, "input")
    test.addLayer(4, 3, "sigmoid")
    test.addLayer(3, "output", "sigmoid")

    test.run([1,1,1])
    output = test.return_output()
    print(output)
