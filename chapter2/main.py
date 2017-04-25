"""Main.py."""
# import os
# import tarfile
# import subprocess
import pprint

# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

from prepared import training_data, test_data

housing_prepared, housing_labels = training_data()
strat_test_set = test_data()


def display_scores(scores, text_=''):
    """Display scores of model."""
    print('\n\n------------------------------------')
    print(text_)
    print('Scoress:', scores)
    print('Mean:', scores.mean())
    print('Standred deviation:', scores.std())
    print('\n\n')

# Training and Evaluating on the Training set
# Better Evaluation Using Cross-Validation


def decisiontreeregressor(prepared, labels):
    """DecisionTreeRegressor."""
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(prepared, labels)
    joblib.dump(tree_reg, 'models/DecisionTreeRegressor.pkl')

    tree_scores = cross_val_score(tree_reg, prepared, labels,
                                  scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-tree_scores)
    display_scores(tree_rmse_scores, text_="DecisionTreeRegressor")


def linearregression(prepared, labels):
    """linearregression."""
    lin_reg = LinearRegression()
    lin_reg.fit(prepared, labels)
    joblib.dump(lin_reg, "models/Linearregression.pkl")

    lin_scores = cross_val_score(lin_reg, prepared, labels,
                                 scoring="neg_mean_squared_error", cv=10)
    line_rmse_scores = np.sqrt(-lin_scores)
    display_scores(line_rmse_scores, text_="LinearRegression")


def randomforestregressor(prepared, labels):
    """RandomForestRegressor."""
    forest_reg = RandomForestRegressor()
    forest_reg.fit(prepared, labels)
    joblib.dump(forest_reg, 'models/RandomForestRegressor.pkl')
    forest_scores = cross_val_score(forest_reg, prepared, labels,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores, text_='RandomForestRegressor')


# Grid Search
def grid_search(prepared, labels):
    """Grid Search."""
    params_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, params_grid, cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(prepared, labels)

    print('best_params_')
    pprint.pprint(grid_search.best_params_)

    print('best_estimator_')
    pprint.pprint(grid_search.best_estimator_)

    cvres = grid_search.cv_results_

    for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
        print(np.sqrt(-mean_score), params)

    return grid_search

if __name__ == "__main__":
    grid_search = grid_search(prepared=housing_prepared, labels=housing_labels)
    joblib.dump(grid_search.best_estimator_, 'models/final.pkl')
