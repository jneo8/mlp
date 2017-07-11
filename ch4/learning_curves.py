"""Learning Curves."""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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


def plot_learning_curves(model, x, y):
    """Learning Curves."""
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(x_train)):
        model.fit(x_train[:m], y_train[:m])
        y_train_predict = model.predict(x_train[:m])
        y_val_predict = model.predict(x_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)


def learning_curves():
    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, x, y)
    plt.axis([0, 80, 0, 3])
    save_img_and_show(name='learning_curves')


def polynomial_10degree():
    polynomial_regression = Pipeline((
        ('poly_features', PolynomialFeatures(degree=10, include_bias=False)),
        ('sgd_reg', LinearRegression()),
    ))
    plot_learning_curves(polynomial_regression, x, y)
    plt.axis([0, 80, 0, 3])
    save_img_and_show('learning_curves_plot')


if __name__ == '__main__':
    high_degree_polynomial_regression()
    learning_curves()
    polynomial_10degree()
