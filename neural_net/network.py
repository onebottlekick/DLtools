import numpy as np
import pickle

from functions import sigmoid, identity, softmax, relu
from typing import List


def activation_match(func_name : str, a : np.ndarray):
    func_list = ['sigmoid', 'identity', 'softmax', 'relu']
    if func_name not in func_list:
        raise ValueError(f'Not a valid function name. your input : {func_name}')
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
        self.activations = []
    
    def add(self, layer):
        if isinstance(layer, dict):
            for name, value in layer.items():
                self.layers[name] = value
        else:
            weight, bias = layer.layer().items()
            self.weights.append(weight[1])
            self.biases.append(bias[1])
            self.activations.append(layer.activation_function())

    def load_and_fit(self, file_path, x : np.ndarray, act_func : List[str]):
        self.activations += act_func 
        with open(file_path, 'rb') as f:
            layers = pickle.load(f)
        self.add(layer=layers)
        for item in self.layers.items():
            try:
                if item[0][0][0] == 'W':
                    self.weights.append(item)
                if item[0][0][0] == 'b':
                    self.biases.append(item)
            except Exception as e:
                print(e, "input example : {'W+idx or b+idx' : np.ndarray}")

        self.weights = sorted(self.weights, key=lambda x:x[0][-1])
        self.biases = sorted(self.biases, key=lambda x:x[0][-1])

        a = np.dot(x, self.weights[0][1]) + self.biases[0][1]
        z = activation_match(self.activations[0], a)
        for i in range(1, len(self.weights) - 1):
            a = np.dot(z, self.weights[i][1]) + self.biases[i][1]
            z = activation_match(self.activations[i], a)
        a = np.dot(z, self.weights[-1][1]) + self.biases[-1][1]
        self.y_pred = activation_match(self.activations[-1], a)

    # TODO model save function 만들어야됨
    def save(self, file_path):
        raise NotImplementedError

    # def prepare(self):
    #     for item in self.layers.items():
    #         try:
    #             if item[0][0][0] == 'W':
    #                 self.weights.append(item)
    #             if item[0][0][0] == 'b':
    #                 self.biases.append(item)
    #         except Exception as e:
    #             print(e, "input example : {'W+idx or b+idx' : np.ndarray}")
    #     self.weights = sorted(self.weights, key=lambda x:x[0][-1])
    #     self.biases = sorted(self.biases, key=lambda x:x[0][-1])

    # TODO 임시 fit method
    def fit(self, x : np.ndarray):
        a = np.dot(x, self.weights[0]) + self.biases[0]
        z = activation_match(self.activations[0], a)
        for i in range(1, len(self.weights) - 1):
            a = np.dot(z, self.weights[i]) + self.biases[i]
            z = activation_match(self.activations[i], a)
        a = np.dot(z, self.weights[-1]) + self.biases[-1]
        self.y_pred = activation_match(self.activations[-1], a)


    def predict(self):
        return self.y_pred