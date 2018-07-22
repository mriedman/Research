import numpy as np
from backpropagation import backpropagation
from ANN import ANN

"""
* Default values
   * episolon = 10**(-8)
   * beta1 = 0.9
   * beta2 = 0.999
   * learning_rate = 0.001
"""

class ADAM():

    def __init__(self, neuralnet, moments_init=None):
        self.ann = neuralnet
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 10**(-8)
        self.learning_rate = 0.02

        if moments_init is None:
            self.biased_first_moment_estimates = []
            self.biased_second_raw_moment_estimates =[]
            self.biased_corrected_first_estimates = []
            self.biased_corrected_second_estimates = []
            self.param_update = []
            self.layer_gradients = []
        for i in range(0, len(self.ann.layers) - 1, 1):
            self.param_update.append(np.zeros_like(self.ann.layers[i].weight))

    def get_gradients(self):
        self.ann = backpropagation(self.ann)
        #self.layer_gradients = []
        for i in range(0, len(self.ann.layers) - 1, 1):
            gradient = self.ann.layers[i].dWeight + self.ann.momentum * self.ann.layers[i].weight
            self.layer_gradients.append(gradient)

    def update_weights(self):
        for i in range(0, len(self.ann.layers) - 1, 1):
            self.ann.layers[i].weight = self.ann.layers[i].weight - self.param_update[i]

    def adam(self, _input, target):

        for epoch in range(self.ann.max_epoch):
            self.ann.run(_input)
            self.ann.target = np.array(target)
            self.get_gradients()

            for layer_gradient in self.layer_gradients:
                init_moment_vector =  np.zeros_like(layer_gradient)

                self.biased_first_moment_estimates.append(init_moment_vector)
                self.biased_second_raw_moment_estimates.append(init_moment_vector)
                self.biased_corrected_first_estimates.append(init_moment_vector)
                self.biased_corrected_second_estimates.append(init_moment_vector)

            for timestep in range(1000):
                for i in range(0, len(self.ann.layers) - 1, 1): #layer_index

                    layer_gradient = self.layer_gradients[i]
                    self.biased_first_moment_estimates[i] = self.beta1 * self.biased_first_moment_estimates[i] + (1 - self.beta2) * layer_gradient
                    self.biased_second_raw_moment_estimates[i] = self.beta2 * self.biased_second_raw_moment_estimates[i] + (1 - self.beta2) * layer_gradient * layer_gradient
                    biased_corrected_first_moment_estimate = self.biased_first_moment_estimates[i] * (1 / (1 - self.beta1**(timestep) + self.epsilon))
                    bias_corrected_second_raw_moment_estimates = self.biased_second_raw_moment_estimates[i] * (1 / (1 - self.beta2**(timestep) + self.epsilon))

                    self.param_update[i] = self.learning_rate * biased_corrected_first_moment_estimate * (1 / (np.sqrt(bias_corrected_second_raw_moment_estimates) + self.epsilon))

                    self.update_weights()

    def get_ann(self):
    
        return self.ann
