import subprocess 
import os
from confs import logconf
from scipy.io import loadmat

logger = logconf.Logger(__file__).logger


mnist_path = 'origin_data/mnist-original.mat'


def download():
    """Down mnist data from another site."""
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
