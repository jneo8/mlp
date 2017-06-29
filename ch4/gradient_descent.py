"""Gradient Descent."""

import numpy as np
from base import save_img_and_show
from confs import logconf


logger = logconf.Logger(__file__).logger


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
