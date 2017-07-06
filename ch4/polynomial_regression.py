"""Polynomial Regression."""
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from confs import logconf
from base import save_img_and_show

logger = logconf.Logger(__file__).logger


def main():

    m = 100
    x = 6 * np.random.rand(m, 1) -3
    y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)
    plt.plot(x, y, 'b.')
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$y$', rotation=0, fontsize=18)
    plt.axis([-1, 3, 0, 10])
    save_img_and_show(name='Genearated nonlinear & noisy dataset')

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly_features.fit_transform(x)
    logger.info('x[0] {}'.format(x[0]))
    logger.info('x_poly[0] {}'.format(x_poly[0]))

    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)
    logger.info('lin_reg.intercept_ {}'.format(lin_reg.intercept_))
    logger.info('lin_reg.coef_ {}'.format(lin_reg.coef_))

    x_new = np.linspace(-3, 3, 100).reshape(100, 1)
    x_new_poly = poly_features.transform(x_new)
    y_new = lin_reg.predict(x_new_poly)
    plt.plot(x, y, 'b.')
    plt.plot(x_new, y_new, 'r-', linewidth=2, label="Predictions")
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$y$', rotation=0, fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([-3, 3, 0, 10])
    save_img_and_show(name="PolynomialRegressionModelPredictions")


if __name__ == '__main__':
    main()
