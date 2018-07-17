from ANN import *

if __name__ == "__main__":
    test = ANN()
    test.addLayer(2, 3, "input")
    test.addLayer(3, 1, "sigmoid")
    test.addLayer(1, "output", "sigmoid")


    test.set_max_epoch(50)
    test.set_desired_error(0.001)
    test.set_momentum(0.05)
    test.set_learning_rate(0.05)
    test.set_erf("MSE")


    _input = [1, 1]
    output = [0]

    test.train(_input, output)

    test.run(_input)
    _output = test.get_output()
    print(_output)
