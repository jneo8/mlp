"""Gradient Descent."""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from base import save_img_and_show, x, x_b, y, x_new, x_new_b
from confs import logconf


logger = logconf.Logger(__file__).logger


def quick_implementation():
    """Gradient Descent quick implementation."""
    eta = 0.1  # learning rate
    n_iterations = 1000

    m = 100
    theta = np.random.randn(2, 1)  # random initialization
    for iteration in range(n_iterations):
        gradients = 2 / m * x_b.T.dot(x_b.dot(theta) - y)
        theta = theta - (eta * gradients)

    logger.info('theta {}'.format(theta))
    logger.info('x_new_b.dot(theta) {}'.format(x_new_b.dot(theta)))


def plot_gradient_descent(theta, eta, theta_path=None):
    """plot_gradient_descent."""
    m = len(x_b)
    plt.plot(x, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = x_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(x_new, y_predict, style)
        gradients = 2 / m * x_b.T.dot(x_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)


def gradient_descent_with_various_learning_rate():
    """Gradient Descent_with_various_learning_rate."""
    theta_path_bgd = []
    rnd.seed(42)
    theta = rnd.randn(2, 1)  # random initialization

    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plot_gradient_descent(theta, eta=0.02)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(132)
    plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
    plt.subplot(133)
    plot_gradient_descent(theta, eta=0.5)

    save_img_and_show(name="gradient_descent_plot")


if __name__ == '__main__':
    quick_implementation()
    gradient_descent_with_various_learning_rate()
