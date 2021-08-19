import numpy as np


class Layer:
    def __init__(self, idx : int, node_size : int, input_size : int, activation : str=None):
        self.idx = idx
        self.node_size = node_size
        self.input_size = input_size
        self.activation = activation
        self.layer_dict = None
    
    def layer(self):
        self.layer_dict = {
            "W" + str(self.idx) : np.random.randn(self.node_size).reshape(self.input_size, -1),
            "b" + str(self.idx) : np.random.randn(int(self.node_size/self.input_size))
        }
        return self.layer_dict

    def output_size(self):
        return self.layer_dict[self.layer_dict['W' + self.idx]][1].shape[-1]

    def activation_function(self):
        if self.activation:
            return self.activation
