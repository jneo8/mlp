"""Linear Regression with TnesorFlow."""
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing


sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from confs import logconf

logger = logconf.Logger(__file__).logger


def main():
    """Main."""
    housing = fetch_california_housing()
    logger.debug(type(housing))
    m, n = housing.data.shape
    logger.debug(f'{m} {n}')
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
    logger.debug(housing_data_plus_bias)
    logger.debug(housing_data_plus_bias.shape)
    x = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='x')
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
    xt = tf.transpose(x)
    logger.debug(xt)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(xt, x)), xt), y)

    with tf.Session() as sess:
        theta_value = theta.eval()
        logger.debug(theta_value)


if  __name__  == '__main__':
    main()
