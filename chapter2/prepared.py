import os
import tarfile
import subprocess
import pprint
from six.moves import urllib

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # ScandardScaler data
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'
datasets_path = BASE_DIR + '/' + HOUSING_PATH


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Download data from url."""
    if not os.path.isdir(datasets_path):
        os.makedirs(datasets_path)
    tgz_path = os.path.join(datasets_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=datasets_path)
    housing_tgz.close()


def load_housing_data(housing_path=datasets_path):
    """Read csv by pandas."""
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


room_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """Custom transform class."""

    def __init__(self, add_bedrooms_per_room=True):
        """init."""
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, x, y=None):
        """Fit, nothing else to do."""
        return self

    def transform(self, x, y=None):
        """transform."""
        rooms_per_household = x[:, room_ix] / x[:, household_ix]
        population_per_household = x[:, population_ix] / x[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = x[:, bedrooms_ix] / x[:, room_ix]
            return np.c_[x, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[x, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Custom Transformer for sklearn to handle pandas Dataframes."""

    def __init__(self, attribute_names):
        """init."""
        self.attribute_names = attribute_names

    def fit(self, x, y=None):
        """fit."""
        return self

    def transform(self, x):
        """transform."""
        return x[self.attribute_names].values


# download data
fetch_housing_data()
print('# download data success!')
print('\n# head()\n')
housing = load_housing_data()
pprint.pprint(housing.head())  # look first 5 rows.
print('\n# info()\n')
pprint.pprint(housing.info())  # get a quick description fo the data.

# Split data

housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

pprint.pprint(housing['income_cat'].value_counts() / len(housing))

# remove income_cat
for set_ in (strat_train_set, strat_test_set):
    set_.drop(['income_cat'], axis=1, inplace=True)

# Discover and Visualize the Data to gain insights
housing = strat_train_set.copy()


# Experimenting with Attribute Combinations

housing["rooms_per_household"] = housing["total_rooms"] / housing['households']
housing['bedrooms_per_room'] = housing["total_bedrooms"] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()


# Transformation Pipelines

housing_num = housing.drop('ocean_proximity', axis=1)
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizer()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)

print('housing_prepared.shape')
pprint.pprint(housing_prepared.shape)

# Training and Evaluating on the Training set
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_labels))

# mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('lin_rmse:', lin_rmse)

print('\n\n---------EndPrepared----------\n\n')


def training_data(housing_prepared=housing_prepared, housing_labels=housing_labels):
    """Return training data."""
    return housing_prepared, housing_labels


def test_data(strat_test_set=strat_test_set):
    """Return test data."""

    return strat_test_set
