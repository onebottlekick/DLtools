import numpy as np



def accuracy_score(y_test : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
    cnt = 0
    if y_pred.ndim != 1:
        y_pred_max = []
        for j in range(len(y_pred)):
            y_pred_max.append(np.argmax(y_pred[j]))
        y_pred = np.array(y_pred_max)

    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            cnt += 1
    return cnt/len(y_test)
