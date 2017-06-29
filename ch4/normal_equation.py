"""The Normal Equation."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from base import save_img_and_show, x, x_b, y, x_new, x_new_b
from confs import logconf

logger = logconf.Logger(__file__).logger


def normal_equation():
    """normal_equation."""
    logger.debug(x)
    logger.debug(y)

    plt.scatter(x, y)
    save_img_and_show(name='randomly_generated_linear_dataset')

    theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
    logger.info(
        '\nEquation y = 4 + 3x0 + Gaussionnoise. theta_best {}'
        .format(theta_best)
    )

    y_predict = x_new_b.dot(theta_best)
    logger.info('y_predict {}'.format(y_predict))

    plt.plot(x_new, y_predict, "r-")
    plt.plot(x, y, "b.")
    plt.axis([0, 2, 0, 15])
    save_img_and_show(name='linear regression model predictions')


def normal_equation_skl():
    """Linear regression model predictions use skl."""
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    logger.info('intercept_ {}'.format(lin_reg.intercept_))
    logger.info('coef_ {}'.format(lin_reg.coef_))
    y_predict = lin_reg.predict(x_new)
    logger.info('predict {}'.format(y_predict))

    plt.plot(x_new, y_predict, "r-")
    plt.plot(x, y, "b.")
    plt.axis([0, 2, 0, 15])
    save_img_and_show(name='linear regression model predictions skl')


if __name__ == '__main__':
    normal_equation()
    normal_equation_skl()
