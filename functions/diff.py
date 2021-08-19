import numpy as np

def numerical_gradient(func, x : np.ndarray, h : float=1e-4 ):
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = func(x)

        x[i] = tmp - h
        fxh2 = func(x)

        grad[i] = (fxh1 -fxh2) / (2 * h)
        x[i] = tmp
    return grad