"""The Normal Equation."""
import subprocess
import numpy as np

from confs import logconf
import matplotlib.pyplot as plt

logger = logconf.Logger(__file__).logger


def save_img_and_show(name):
    """save_img_and_show."""
    plt.savefig('img/{}'.format(name))
    plt.clf()
    subprocess.call(['catimg', '-f', 'img/{}.png'.format(name)])


def normal_equation():
    """normal_equation."""
    x = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1)

    logger.debug(x)
    logger.debug(y)

    plt.scatter(x, y)
    save_img_and_show(name='p1')

if __name__ == '__main__':
    normal_equation()
