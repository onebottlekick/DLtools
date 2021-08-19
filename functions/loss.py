import numpy as np

def sum_sqares_error(pred, y):
    return 0.5 * np.sum((pred -y) ** 2)

def cross_entropy_error(pred, y, delta=1e-7):
    if pred.ndim == 1:
        pred = pred.reshape(1, pred.size)
        y = y.reshape(1, y.size)
    
    batch_size = pred.shape[0]
    return -np.sum(np.log(pred[np.arange(batch_size), y] + delta)) / batch_size