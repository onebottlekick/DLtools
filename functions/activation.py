import numpy as np

def step_function(x : np.ndarray) -> np.ndarray:
    y = x > 0
    return y.astype(np.int)

def sigmoid(x:np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def relu(x : np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

# data need to be normalized
def softmax(x : np.ndarray) -> np.ndarray:
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y