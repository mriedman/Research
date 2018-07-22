from ANN import *
from Layer import *
import numpy as np



def backpropagation(neuralnet):

    #output layer delta
    output_z = neuralnet.layers[len(neuralnet.layers) - 1].get_z_values() #column vector
    output_dz = neuralnet.layers[len(neuralnet.layers) - 1].activation_function_derivative(output_z) #column vector
    delta_output = neuralnet.loss_gradient_wrt_output() #* output_dz #column vector
    neuralnet.layers[len(neuralnet.layers) - 1].set_delta(delta_output) #column vector

    #get hidden layer deltas
    for i in range(len(neuralnet.layers) - 2, 0, -1):
        z_vals = neuralnet.layers[i].get_z_values()
        a_val_derivatives = neuralnet.layers[i].activation_function_derivative(z_vals)
        hidden_delta = np.matmul(np.transpose(neuralnet.layers[i].weight), neuralnet.layers[i+1].delta) * a_val_derivatives
        neuralnet.layers[i].set_delta(hidden_delta)

    #get delta weight
    for i in range(0, len(neuralnet.layers) - 1, 1):
        if i == 0:
            a_vals = neuralnet._input
        else:
            a_vals = neuralnet.layers[i].get_activation_values()

        neuralnet.layers[i].dWeight = neuralnet.layers[i].dWeight + np.matmul(neuralnet.layers[i+1].delta, np.transpose(a_vals))
        #neuralnet.layers[i].dBias = neuralnet.layers[i].dBias + np.matmul(neuralnet.layers[i+1].delta, np.transpose(a_vals))
        neuralnet.layers[0].dBias = neuralnet.layers[0].dBias + neuralnet.layers[1].delta


    return neuralnet
