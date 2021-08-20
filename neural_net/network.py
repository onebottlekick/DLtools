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
layer1 = Layer(idx=1, node_size=6, input_size=2, activation='sigmoid')
layer2 = Layer(idx=2, node_size=6, input_size=3, activation='sigmoid')
layer3 = Layer(idx=3, node_size=4, input_size=2, activation='identity')
model.add(layer1)
model.add(layer2)
model.add(layer3)
model.fit(x)
prediction = model.predict()
"""
class NeuralNet:
    def __init__(self):
        self.parameters = {}
        self.weights = []
        self.biases = []
        self.y_pred = None
        self.activations = []
    
    def add(self, layer):
        if isinstance(layer, dict):
            for name, value in layer.items():
                self.parameters[name] = value
        else:
            weight, bias = layer.layer().items()
            self.parameters[weight[0]] = weight[1]
            self.parameters[bias[0]] = bias[1]
            self.weights.append(weight[1])
            self.biases.append(bias[1])
            self.activations.append(layer.activation_function())

    # Usage:
    # model = NeuralNet()
    # model.load_and_fit('sample_weight.pkl', X_test, ['sigmoid', 'sigmoid', 'softmax'])
    # y_pred = model.predict()
    def load(self, file_path, x : np.ndarray, act_func : List[str]):
        self.activations += act_func 
        with open(file_path, 'rb') as f:
            layers = pickle.load(f)
        self.add(layer=layers)
        for item in self.parameters.items():
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

    # Usage:
    # model.save({filePath})
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.parameters, f)


    # TODO have to update (gradient descent)
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