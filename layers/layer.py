import numpy as np


class Layer:
    def __init__(self,  node_size : int, input_size : int, names=None, activation : str=None):
        self.names = names
        self.node_size = node_size
        self.input_size = input_size
        self.activation = activation
        self.layer_list = None
    
    def layer(self):
        self.layer_dict = {
            self.names[0] : np.random.randn(self.node_size).reshape(self.input_size, -1),
            self.names[1] : np.random.randn(int(self.node_size/self.input_size))
        }

        return self.layer_dict

    def output_size(self):
        return self.layer_dict[self.names[0]][1].shape[-1]

    def activation_function(self):
        if self.activation:
            return self.activation
