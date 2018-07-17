from ANN import *

if __name__ == "__main__":
    test = ANN()
    test.createFromArrays([2,3,1],["sigmoid","sigmoid"])

    test.set_max_epoch(50)
    test.set_desired_error(0.001)
    test.set_momentum(0.05)
    test.set_learning_rate(0.05)
    test.set_erf("MSE")


    _input = [1, 1]
    output = [0]

    test.train(_input, output)


