"""Test."""
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from read import read
from confs import logconf
logger = logconf.Logger(__file__).logger


class Test():
    """Tset class."""

    def __init__(self):
        """Init."""
        mnist = read()
    
        self.x, self.y = mnist['data'], mnist['target']
    
        # show shape
        logger.info('x.shape {}'.format(self.x.shape))
        logger.info('y.shape {}'.format(self.y.shape))
        
        # split train&test data set
        self.x_train, self.x_test = self.x[:60000], self.x[60000:] 
        self.y_train, self.y_test = self.y[:60000], self.y[60000:] 
        shuffle_index = np.random.permutation(60000)
        self.x_train, y_train = self.x_train[shuffle_index], self.y_train[shuffle_index] 

    @property
    def p1(self):
        some_digit = self.x[36000]
        some_digit_image = some_digit.reshape(28, 28)
        plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
        plt.axis('off')
        plt.savefig('img/c2p1')
        subprocess.call(['catimg', '-f', 'img/c2p1.png'])
        logger.info('y[36000] : {}'.format(self.y[36000]))


if __name__ == '__main__':
    t = Test()
    t.p1
