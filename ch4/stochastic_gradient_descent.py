"""Stochastic Gradient descent."""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from base import save_img_and_show, x, x_b, y, x_new, x_new_b
from confs import logconf


logger = logconf.Logger(__file__).logger


def sgd_simple_learning_schedule():
    """SGD use simple_learning_schedule."""
    n_epochs = 50
    t0, t1 = 5, 50  # learning schedule hyperparameters

    def learning_schedule(t):
        return t0 / (t + t1)

    theta = np.random.randn(2, 1)  # random initialization
    m = len(x_b)
    rnd.seed(42)

    for epoch in range(n_epochs):
        for i in range(m):
            if epoch == 0 and i < 20:
                y_predict = x_new_b.dot(theta)
                style = "b-" if i > 0 else "r--"
                plt.plot(x_new, y_predict, style)
            random_index = np.random.randint(m)
            xi = x_b[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradients

    logger.info('theta :{}'.format(theta))

    plt.plot(x, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    save_img_and_show(name="sgd_plot")


def sgd_skl():
    """SGD with skl."""
    sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
    sgd_reg.fit(x, y.ravel())
    logger.debug('sgd_reg.intercept_ {}'.format(sgd_reg.intercept_))
    logger.debug('sgd_reg.coef_ {}'.format(sgd_reg.coef_))


if __name__ == '__main__':
    sgd_simple_learning_schedule()
    sgd_skl()
