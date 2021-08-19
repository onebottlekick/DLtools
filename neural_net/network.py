import numpy as np

from functions.activation import sigmoid, identity
from typing import Dict


"""
Usage:
net = NeuralNet()
net.add(np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]), np.array([0.1, 0.2, 0.3]))
net.add(np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]), np.array([0.1, 0.2]))
net.add(np.array([[0.1, 0.3], [0.2, 0.4]]), np.array([0.1, 0.2]))
print(net.forward(np.array([1.0, 0.5])))
"""
class NeuralNet:
    def __init__(self):
        self.layers = {}
        self.weights = []
        self.biases = []
    
    def add(self, parameters : Dict[str, np.ndarray]):
        for name, value in parameters.items():
            self.layers[name] = value

    def forward(self, x : np.ndarray) -> np.ndarray:
        for layer in self.layers.items():
            try:
                if layer[0][0][0] == 'W':
                    self.weights.append(layer)
                if layer[0][0][0] == 'b':
                    self.biases.append(layer)
            except Exception as e:
                print(e)

        a = np.dot(x, self.weights[0][1]) + self.biases[0][1]
        z = sigmoid(a)
        for i in range(1, len(self.weights) - 1):
            a = np.dot(z, self.weights[i][1]) + self.biases[i][1]
            z = sigmoid(a)
        a = np.dot(z, self.weights[-1][1]) + self.biases[-1][1]
        y = identity(a)
        return y