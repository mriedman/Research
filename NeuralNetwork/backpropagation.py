from ANN import ANN
from Layer import Layer
import numpy as np

def backpropagation(neuralnet, target):

    #output layer
    output_z = neuralnet.layers[len(neuralnet.layers) - 1].return_z_values()
    output_dz = neuralnet.activation_function_derivative(output_z)
    delta_output = neuralnet.loss_gradient() * np.transpose(output_dz)
    neuralnet.layers[len(neuralnet.layers) - 1].set_delta(delta_output)


    #hidden layer
    for i in range(len(neuralnet.layers) - 2, 1, -1):
        hidden_z = neuralnet.layers[i].return_z_values()
        hidden_dz = neuralnet.activation_function_derivative(hidden_z)
        delta_hidden = np.transpose(np.matmul(np.transpose(neuralnet.layers[i].weight), np.transpose(neuralnet.layers[i+1].delta))) * hidden_dz
        neuralnet.layers[i].set_delta(_delta_hidden)


    #input layer
    neuralnet.layers[0].dWeights = neuralnet.layers[0].dWeights + np.matmul(np.transpose(neuralnet.layers[1].delta), neuralnet.input)
    neuralnet.layers[0].dBias = neuralnet.layers[0].dWeights + np.matmul(np.transpose(neuralnet.layers[1].delta), neuralnet.input)

    #non_input layer dWeights
    for i in range(i, len(neuralnet.layers) - 1, 1):
        layer_a_values = neuralnet.layers[i].get_activation_values()
        neuralnet.layers[i].dWeight = neuralnet.layers[i].delta_weights + np.matmul(np.transpose(neuralnet.layers[i+1].delta), layer_a_values)
        neuralnet.layers[i].dBias = neuralnet.layers[i].dBias + np.matmul(neuralnet.layers[i+1].delta, np.transpose(layer_a_values)

    return neuralnet
