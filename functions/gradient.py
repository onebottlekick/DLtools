from .diff import numerical_gradient

def gradient_descent(func, x, learning_rate=0.01, step_num=100):
    for i in range(step_num):
        grad = numerical_gradient(func, x)
        x -= learning_rate * grad
    return x