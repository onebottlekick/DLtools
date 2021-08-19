import numpy as np

import matplotlib.pyplot as plt


def plot_activation_func(func) -> None:
    x = np.arange(-5, 5.0, 0.1)
    y = func(x)
    plt.plot(x, y)
    plt.ylim(-1.1, 5.1)
    plt.show()

def plot_img(img : np.ndarray):
    img_plot = plt.imshow(img)
    plt.show()