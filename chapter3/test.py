"""Test."""
import subprocess
import matplotlib
import matplotlib.pyplot as plt
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
