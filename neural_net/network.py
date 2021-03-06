import numpy as np

from functions.activation import sigmoid, identity, softmax, relu
from typing import Dict, List


def activation(func_name : str, a : np.ndarray):
    func_list = ['sigmoid', 'identity', 'softmax', 'relu']
    if func_name not in func_list:
        raise ValueError('not a valid function name')
    if func_name == 'sigmoid':
        return sigmoid(a)
    elif func_name == 'identity':
        return identity(a)
    elif func_name == 'softmax':
        return softmax(a)
    elif func_name == 'relu':
        return relu(a)

        
"""
Usage:
model = NeuralNet()
parameters = {
    'W1' : np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
    'W2' : np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
    'W3' : np.array([[0.1, 0.3], [0.2, 0.4]]),
    'b1' : np.array([0.1, 0.2, 0.3]),
    'b2' : np.array([0.1, 0.2]),
    'b3' : np.array([0.1, 0.2])
    }
model.add(parameters=parameters)
model.fit(np.array([1.0, 0.5]), ['sigmoid', 'sigmoid', 'identity'])
y_pred = model.predict()
"""
class NeuralNet:
    def __init__(self):
        self.layers = {}
        self.weights = []
        self.biases = []
        self.y_pred = None
    
    def add(self, parameters : Dict[str, np.ndarray]):
        for name, value in parameters.items():
            self.layers[name] = value

    def fit(self, x : np.ndarray, act_func : List[str]) -> np.ndarray:
        for layer in self.layers.items():
            try:
                if layer[0][0][0] == 'W':
                    self.weights.append(layer)
                if layer[0][0][0] == 'b':
                    self.biases.append(layer)
            except Exception as e:
                print(e, "input example : {'W+idx or b+idx' : np.ndarray}")
        try:
            self.weights = sorted(self.weights, key=lambda x:x[0][-1])
            self.biases = sorted(self.biases, key=lambda x:x[0][-1])
            a = np.dot(x, self.weights[0][1]) + self.biases[0][1]
            z = activation(act_func[0], a)
            for i in range(1, len(self.weights) - 1):
                a = np.dot(z, self.weights[i][1]) + self.biases[i][1]
                z = activation(act_func[i], a)
            a = np.dot(z, self.weights[-1][1]) + self.biases[-1][1]
            self.y_pred = activation(act_func[-1], a)
        except Exception as e:
            print(e)

    def predict(self):
        return self.y_pred