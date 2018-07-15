from ANN import ANN
from Layer import Layer
import numpy as np

def backpropagation(neuralnet):

    #neuralnet = ann
    #output layer
    output_z = neuralnet.layers[len(neuralnet.layers) - 1].get_z_values()
    output_dz = neuralnet.layers[len(neuralnet.layers) - 1].activation_function_derivative(output_z)
    delta_output = neuralnet.loss_gradient() * output_dz
    neuralnet.layers[len(neuralnet.layers) - 1].set_delta(delta_output)


    #hidden layer
    for i in range(len(neuralnet.layers) - 2, 0, -1):
        hidden_z = neuralnet.layers[i].get_z_values()
        hidden_dz = neuralnet.layers[i].activation_function_derivative(hidden_z)
        delta_hidden = np.transpose(np.matmul(np.transpose(neuralnet.layers[i].weight), np.transpose(neuralnet.layers[i+1].delta))) * hidden_dz
        neuralnet.layers[i].set_delta(delta_hidden)

    #print(neuralnet.layers[1].delta)
    #print(neuralnet.input)

    #input layer
    """print("dweight: " + str(neuralnet.layers[0].dWeight.shape))
    print("delta: " + str(neuralnet.layers[1].delta.shape))
    print("input: " + str(neuralnet.input.shape))"""
    neuralnet.layers[0].dWeight = neuralnet.layers[0].dWeight + np.matmul(np.transpose(neuralnet.layers[1].delta), neuralnet.input)
    """print("dBias: " + str(neuralnet.layers[0].dBias.shape))
    print("delta: " + str(neuralnet.layers[1].delta.shape))
    print("input: " + str(neuralnet.input.shape))"""
    #neuralnet.layers[0].dBias = neuralnet.layers[0].dBias + np.matmul(np.transpose(neuralnet.layers[1].delta), neuralnet.input)
    neuralnet.layers[0].dBias = neuralnet.layers[0].dBias + neuralnet.layers[1].delta
    #print(len(neuralnet.layers))
    #non_input layer dWeights
    for i in range(1, len(neuralnet.layers) - 1, 1):
        layer_a_values = neuralnet.layers[i].get_activation_values()
        #print(neuralnet.layers[i+1].delta.shape)
        #print(layer_a_values.shape)
        #print(neuralnet.layers[i].dWeight.shape)
        neuralnet.layers[i].dWeight = neuralnet.layers[i].dWeight+ np.matmul(neuralnet.layers[i+1].delta, layer_a_values)
        #neuralnet.layers[i].dBias = neuralnet.layers[i].dBias + np.matmul(np.transpose(neuralnet.layers[i+1].delta), layer_a_values)
        neuralnet.layers[i].dBias = neuralnet.layers[i].dBias + neuralnet.layers[i].delta

    print("backprop complete")
    return neuralnet
