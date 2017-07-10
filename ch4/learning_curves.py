"""Learning Curves."""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from base import save_img_and_show

from confs import logconf


logger = logconf.Logger(__file__).logger

m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)
x_new = np.linspace(-3, 3, 100).reshape(100, 1)


def high_degree_polynomial_regression():
    """High-degree Polynomial Regression."""
    for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
        polybig_features = PolynomialFeatures(
            degree=degree,
            include_bias=False
        )
        std_scaler = StandardScaler()
        lin_reg = LinearRegression()
        polynomial_regression = Pipeline((
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ))
        polynomial_regression.fit(x, y)
        y_newbig = polynomial_regression.predict(x_new)
        plt.plot(x_new, y_newbig, style, label=str(degree), linewidth=width)

    plt.plot(x, y, "b.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([-3, 3, 0, 10])
    save_img_and_show(name='high_degree_polynomials_plot')

if __name__ == '__main__':
    high_degree_polynomial_regression()
