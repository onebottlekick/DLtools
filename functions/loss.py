import numpy as np

def accuracy(y_test : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
    return y_test == y_pred

def accuracy_score(y_test : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
    sum(accuracy(y_test, y_pred)) / len(y_test)