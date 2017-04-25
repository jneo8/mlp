"""Final.py."""
import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.externals import joblib

from prepared import test_data, full_pipeline
from main import display_scores

strat_test_set = test_data()


def final(final_model, test_data):
    """final."""
    x_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    x_test_prepared = full_pipeline.transform(x_test)
    # final_predictions = final_model.predict(x_test_prepared)
    # final_mse = mean_squared_error(y_test, final_predictions)
    # final_rmse = np.sqrt(final_mse)
    # pprint.pprint(final_predictions)
    # pprint.pprint(final_rmse)

    scores = cross_val_score(final_model, x_test_prepared, y_test, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)

final_model = joblib.load('models/final.pkl')
final(final_model=final_model, test_data=strat_test_set)
