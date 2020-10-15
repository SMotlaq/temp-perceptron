import numpy as np
import math

class perceptron():

    _weights           = 0
    _outputs           = 0
    _number_of_inputs  = 0
    _number_of_outputs = 0

    def __init__(self, number_of_inputs, number_of_outputs):
        self._weights = np.random.rand(number_of_inputs+1,number_of_outputs)
        self._number_of_inputs  = number_of_inputs
        self._number_of_outputs = number_of_outputs

    def update(self, input_vector):
        perceptron_input = np.append(input_vector, [-1]).reshape(1,self._number_of_inputs+1)
        f_net = np.dot(perceptron_input, self._weights)
        self._outputs = self.sig(f_net)
        return self._outputs

    def train(self, number_of_data, input_vectors, desired_outputs):
        eta = 0.9
        E_max = 0.001
        E = np.ones([np.shape(input_vectors)[0],1])
        while((E>E_max).any()):
            E = 0.0
            for i in range(number_of_data):
                y = input_vectors[i]
                d = desired_outputs[i]
                o = self.update(y)
                self._weights = self._weights + 0.5 * eta * (d - o) * (1 - o * o) * np.append(y,[-1]).reshape(self._number_of_inputs+1,1)
                E = E + 0.5 * (d - o) * (d - o)
            print(E)
        return 1

    def sig(self, input):
        out = np.zeros((input.shape[0],input.shape[1]))
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                out[i][j] = 1/(1+math.exp(-input[i][j]))
        return out

myClassifier = perceptron(2, 2)
x = np.array([[0,0],[0,1],[1,0],[1,1]])
desired = np.array([[0,0],[0,1],[0,1],[1,1]])
myClassifier.train(4, x, desired)
print('==== TEST ====')
new_input = np.array([[0,0],[0,1],[1,0],[1,1]])
for i in range(4):
    print(myClassifier.update(new_input[i]))

print(myClassifier._weights)
