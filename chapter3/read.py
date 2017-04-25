import subprocess 
import os
from confs import logconf
from scipy.io import loadmat

logger = logconf.Logger(__file__).logger


mnist_path = 'origin_data/mnist-original.mat'


def download():
    """Down mnist data from another site."""
    """
    "The book p.79 have error.
    "https://github.com/login/oauth/authorize?client_id=7e0a3cd836d3e544dbd9&redirect_uri=https%3A%2F%2Fgist.github.com%2Fauth%2Fgithub%2Fcallback%3Freturn_to%3Dhttps%253A%252F%252Fgist.github.com%252Fyoungsoul%252Ffc69665c5d08e189c57c0db0e93017a6&response_type=code&state=9b385430ee7cd1a75ca91c1d1cb6c565111f6b81e54a71f42ae9b22035241b9b
    """
    subprocess.call([
        'wget',
        'https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat', 
        '-P',
        'origin_data/'
    ])
    logger.info('Download success!')


def read():
    """Return mnist data."""
    if os.path.exists(mnist_path):
        logger.info('mnist-original.mat file exists.')
    else:
        logger.info('start download!')
        download()
    mnist_raw = loadmat(mnist_path)
    mnist = {
        'data': mnist_raw["data"].T,
        'target': mnist_raw['label'][0],
        'COL_NAMES': ["label", "data"],
        'DESCR': 'mldata.org dataset: mnist-original',
    }
    return mnist

if __name__ == '__main__':
    logger.info('read.py')
    mnist = read()
    print(mnist)
