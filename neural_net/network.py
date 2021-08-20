import numpy as np
import pickle

from functions import sigmoid, identity, softmax, relu
from typing import List

# connect string to activation function
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
-----------------------------------------------------------------------
Usage:
------------------------------------------------------------------------
model = NeuralNet()
layer1 = Layer(idx=1, node_size=6, input_size=2, activation='sigmoid')
layer2 = Layer(idx=2, node_size=6, input_size=3, activation='sigmoid')
layer3 = Layer(idx=3, node_size=4, input_size=2, activation='identity')
model.add(layer1)
model.add(layer2)
model.add(layer3)
model.fit(x)
prediction = model.predict()
-------------------------------------------------------------------------
"""
class NeuralNet:
    def __init__(self):
        self.parameters = {}
        self.weights = []
        self.biases = []
        self.y_pred = None
        self.activations = []
        self.compute= {}


    """
    ----------------------------------------------------------------------
    Usage:
    ----------------------------------------------------------------------
    <case 1>
    parameters = {
    'W1' : np.array([0.1, 0.3, 0.5, 0.2, 0.4, 0.6]).reshape(2, 3),
    'b1' : np.array([0.1, 0.2, 0.3]),
    'W2' : np.array([0.1, 0.4, 0.2, 0.5, 0.3, 0.6]).reshape(3, 2),
    'b2' : np.array([0.1, 0.2]),
    'W3' : np.array([0.1, 0.3, 0.2, 0.4]).reshape(2, 2),
    'b3' : np.array([0.1, 0.2]),
    'activations' : ['sigmoid', 'sigmoid', 'identity']
    }
    model.add(parameters)
    ------------------------------------------------------------------------
    <case 2>
    layer1 = Layer(idx=1, node_size=6, input_size=2, activation='sigmoid')
    layer2 = Layer(idx=2, node_size=6, input_size=3, activation='sigmoid')
    layer3 = Layer(idx=3, node_size=4, input_size=2, activation='identity')
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    --------------------------------------------------------------------------
    """
    def add(self, layer):
        if isinstance(layer, dict):
            for name, value in layer.items():
                self.parameters[name] = value
        else:
            weight, bias = layer.layer().items()
            self.parameters[weight[0]] = weight[1]
            self.parameters[bias[0]] = bias[1]
            # self.weights.append(weight[1])
            # self.biases.append(bias[1])
            self.activations.append(layer.activation_function())
            self.parameters['activations'] = self.activations


    """
    ------------------------------------------------------------------------------------
    Usage:
    model = NeuralNet()
    model.load('sample_weight.pkl', (['sigmoid', 'sigmoid', 'softmax']))
    y_pred = model.predict()
    --------------------------------------------------------------------------------------
    """
    def load(self, file_path, act_func : List[str]=None):
        if act_func:
            self.activations += act_func
            self.parameters['activations'] = self.activations
        with open(file_path, 'rb') as f:
            parameters = pickle.load(f)
        self.add(layer=parameters)


    """
    --------------------------------------------------------------------------
    Usage:
    --------------------------------------------------------------------------
    model.save({filePath})
    --------------------------------------------------------------------------
    """
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.parameters, f)


    # TODO have to update (gradient descent)
    def fit(self, x : np.ndarray):

        for idx in range(1, int(len(self.parameters)/2)+1):
            if idx == 1:
                self.compute['a' + str(idx)] = np.dot(x, self.parameters['W' + str(idx)]) + self.parameters['b' + str(idx)]
                self.compute['z' + str(idx)] = activation_match(self.parameters['activations'][idx-1], self.compute['a' + str(idx)])
            else:
                self.compute['a' + str(idx)] = np.dot(self.compute['z' + str(idx-1)], self.parameters['W' + str(idx)]) + self.parameters['b' + str(idx)]
                self.compute['z' + str(idx)] = activation_match(self.parameters['activations'][idx-1], self.compute['a' + str(idx)])
        self.y_pred = self.compute['z' + str(idx)]

    def predict(self):
        return self.y_pred