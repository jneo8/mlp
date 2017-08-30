"""Gradient Descenti with tensorflow example."""
import os
import sys
import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from confs import logconf  # noqa
logger = logconf.Logger(__file__).logger

N_EPOCHS = 1000
LEARNING_RATE = 0.01


def main():
    """main."""
    # define epochs number and learning rate
    n_epochs = N_EPOCHS
    learning_rate = LEARNING_RATE

    # Get data
    housing = fetch_california_housing()
    m, n = housing.data.shape

    # Use Batch Gradient Descent
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
    x = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='x')
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

    #################################################
    # random_uniform() function create a node in the graph that will generate
    # a tensor containing random values, given its shape and value range,
    # much like Numpy's rand() function.
    #################################################
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(x, theta, name='predictions')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
    gradients = 2 / m * tf.matmul(tf.transpose(x), error)

    #################################################
    # assign() function
    # creates a node that will assign a new value to a variable.
    # In this case, it implements the Batch Gradient Descent
    # theta(next step) = theta - gradients
    #################################################
    training_op = tf.assign(theta, theta - learning_rate * gradients)
    init = tf.global_variables_initializer()

    # tf session
    with tf.Session() as sess:
        sess.run(init)
        #################################################
        # The loop executes the training step n_epochs times
        #################################################
        for epoch in range(n_epochs + 1):
            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch}, MSE = {mse.eval()}')  # noqa
                logger.debug(f'theta \n{theta.eval()}')
                logger.debug(f'error \n{error.eval()}')
                logger.debug(f'gradients \n{gradients.eval()}')
            sess.run(training_op)

        best_theta = theta.eval()


if __name__ == '__main__':
    main()
