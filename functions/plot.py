from typing import Callable
import numpy as np

import matplotlib.pyplot as plt


def plot_activation_func(func) -> None:
    x = np.arange(-5, 5.0, 0.1)
    y = func(x)
    plt.plot(x, y)
    plt.ylim(-1.1, 5.1)
    plt.show()