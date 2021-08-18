import numpy as np

# Gates
def AND(x1, x2):
    x = np.array([x1, x2])    
    w = np.array([0.5, 0.5])
    b = -0.7
    signal = np.sum(w*x) + b
    if signal <= 0:
        return 0
    return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    signal = np.sum(w*x) + b
    if signal <= 0:
        return 0
    return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    signal = np.sum(w*x) + b
    if signal <= 0:
        return 0
    return 1
    
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))