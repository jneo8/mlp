"""Polynomial Regression."""
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from confs import logconf
from base import save_img_and_show


logger = logconf.Logger(__file__).logger


def main():

    m = 100
    x = 6 * np.random.rand(m, 1) -3
    y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)
    plt.plot(x, y, 'b.')
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$y$', rotation=0, fontsize=18)
    plt.axis([-1, 3, 0, 10])
    save_img_and_show(name='Genearated nonlinear & noisy dataset')


if __name__ == '__main__':
    main()
