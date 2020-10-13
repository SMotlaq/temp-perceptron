import numpy as np
import math

class perceptron():

    _weights = 0
    _output  = 0
    _number_of_inputs = 0

    def __init__(self, number_of_inputs):
        self._weights = np.random.rand(number_of_inputs+1,1)
        self._number_of_inputs = number_of_inputs

    def update(self, input_vector):
        n = 0
        for i in range(self._number_of_inputs):
            n = n + input_vector[i]*self._weights[i][0]
        n = n - 1*self._weights[self._number_of_inputs][0]
        self._output = 1/(1+math.exp(-n))
        return self._output

    def train(self, number_of_data, input_vectors, desired_outputs):
        eta = 0.9
        E_max = 0.001
        E = 1
        while(E>E_max):
            E = 0.0
            for i in range(number_of_data):
                y = input_vectors[:,i]
                d = desired_outputs[i]
                o = self.update(y)
                self._weights = self._weights + 0.5 * eta * (d - o) * (1 - o * o) * np.append(y,[-1]).reshape(self._number_of_inputs+1,1)
                E = E + 0.5 * (d - o) * (d - o)
            print(E)
        return 1

myClassifier = perceptron(2)
x = np.array([[3, 3, 4, 4],[3, 4, 3, 4]])
desired = [0, 0, 0, 1]
myClassifier.train(4, x, desired)
print('==== TEST ====')
new_input = np.array([[5, 0, 3, 4],[4, 0, 3, 4]])
for i in range(4):
    print(myClassifier.update(new_input[:,i]))

print(myClassifier._weights)
