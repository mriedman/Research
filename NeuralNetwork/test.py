from ANN import *
from adam import *

if __name__ == "__main__":
    test = ANN()
    test.createFromArrays([2,2,1],["linear","sigmoid"])

    test.set_max_epoch(50)
    test.set_desired_error(0.00000001)
    test.set_momentum(0.05)
    test.set_learning_rate(0.05)
    test.set_erf("MSE")


    #_input =[[1, 0], [0, 1]], [1,1], [0, 0]]
    #output = [[1], [1]], [0], [0]]

    #test.batch_train(_input, output, 1)

    #test adam

    _input = [1, 2]
    target = [0]
    test.run(_input)
    print(test.output)

    test = ADAM(test)
    test.adam(_input, target)
    test = test.get_ann()

    test.run(_input)
    print(test.output)
