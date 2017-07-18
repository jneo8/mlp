"""Base function & attribute."""
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np


from confs import logconf

logger = logconf.Logger(__file__).logger


def save_img_and_show(name):
    """save_img_and_show."""
    plt.savefig('img/{}'.format(name))
    plt.clf()
    logger.info(name)
    subprocess.call(['catimg', '-f', 'img/{}.png'.format(name)])
    os.remove('img/{}.png'.format(name))


x = 2 * np.random.rand(100, 1)
x_b = np.c_[np.ones((100, 1)), x]
y = 4 + 3 * x + np.random.randn(100, 1)
x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2, 1)), x_new]  # add x0 = 1 to each instance

