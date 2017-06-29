"""Base function."""
import os
import subprocess
import matplotlib.pyplot as plt

from confs import logconf

logger = logconf.Logger(__file__).logger


def save_img_and_show(name):
    """save_img_and_show."""
    plt.savefig('img/{}'.format(name))
    plt.clf()
    logger.info(name)
    subprocess.call(['catimg', '-f', 'img/{}.png'.format(name)])
    os.remove('img/{}.png'.format(name))
