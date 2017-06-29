"""Gradient Descent."""
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from confs import logconf


logger = logconf.Logger(__file__).logger


def save_img_and_show(name):
    """save_img_and_show."""
    plt.savefig('img/{}'.format(name))
    plt.clf()
    logger.info(name)
    subprocess.call(['catimg', '-f', 'img/{}.png'.format(name)])
    os.remove('img/{}.png'.format(name))


def quick_implementation():
    """Gradient Descent quick implementation."""
    x = 2 * np.random.rand(100, 1)
    x_b = np.c_[np.ones((100, 1)), x]
    y = 4 + 3 * x + np.random.randn(100, 1)

    eta = 0.1  # learning rate
    n_iterations = 1000

    m = 100
    theta = np.random.randn(2, 1)  # random initialization
    for iteration in range(n_iterations):
        gradients = 2 / m * x_b.T.dot(x_b.dot(theta) - y)
        theta = theta - (eta * gradients)

    logger.info('theta {}'.format(theta))


if __name__ == '__main__':
    quick_implementation()
