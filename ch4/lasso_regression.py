"""Lasso Regression"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from base import save_img_and_show

from confs import logconf

logger = logconf.Logger(__file__).logger


rnd.seed(42)
m = 20
x = 3 * rnd.rand(m, 1)
y = 1 + 0.5 * x + rnd.randn(m, 1) / 1.5
x_new = np.linspace(0, 3, 100).reshape(100, 1)


def plot_model(model_class, polynomial, alphas, **model_kargs):
    """plot_model."""
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(
            alpha, **model_kargs
        ) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline((
                (
                    "poly_features",
                    PolynomialFeatures(degree=10, include_bias=False)
                ),
                ("std_scaler", StandardScaler()),
                ("regul_reg", model),
            ))
        model.fit(x, y)
        y_new_regul = model.predict(x_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(
            x_new,
            y_new_regul,
            style,
            linewidth=lw,
            label=r"$\alpha = {}$".format(alpha)
        )
    plt.plot(x, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])


def main():
    """Main."""
    logger.info('Lasso Regression')
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1))
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    # Show different alpha of lasso regression
    plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), tol=1)

    save_img_and_show(name="lasso_regression_plot")
    plt.show()


if __name__ == '__main__':
    main()
