import os
import tarfile
from six.moves import urllib
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import subprocess
import numpy as np

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


# download data
#fetch_housing_data()
print('# download data success!')


# head()

print('\n# head()\n')
housing = load_housing_data()
pprint.pprint(housing.head())  # look first 5 rows.


# info()

print('\n# info()\n')
pprint.pprint(housing.info())  # get a quick description fo the data.

# value_count()

print('\n# ocean_proximity value_counts()\n')

pprint.pprint(housing['ocean_proximity'].value_counts())

# describe()

print('\n# describe()\n')

pprint.pprint(housing.describe())


# hist()

print('\n# hist()\n')

housing.hist(bins=50, figsize=(20, 15))
plt.savefig('p1')
subprocess.call(['catimg', '-f', 'p1.png'])


#split data

# from testset import split_train_test_by_id
# print("\nSplit data.\n")
# housing_with_id = housing.reset_index()  # adds an 'index' colume
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
# print(len(train_set), "train + ", len(test_set), "test")

# split data use sklearn learn
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# print(len(train_set), "train + ", len(test_set), "test")


# income_cat
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

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

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.savefig('p2')
subprocess.call(['catimg', '-f', 'p2.png'])

housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=housing['population'] / 100,
    label="population",
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
)
plt.savefig('p3')
subprocess.call(['catimg', '-f', 'p3.png'])

# Look for Correlations


# way 1
# corr_matrix = housing.corr()
# pprint.pprint(corr_matrix['median_house_value'].sort_values(ascending=False))
# end way 1

# way 2

from pandas.tools.plotting import scatter_matrix

attribute = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attribute], figsize=(12, 8))
plt.savefig('p4')
subprocess.call(['catimg', '-f', 'p4.png'])

housing.plot(kind='scatter', x='median_income', y="median_house_value", alpha=0.1)
plt.savefig('p5.png')
subprocess.call(['catimg', '-f', 'p5.png'])


# Experimenting with Attribute Combinations

housing["rooms_per_household"] = housing["total_rooms"] / housing['households']
housing['bedrooms_per_room'] = housing["total_bedrooms"] / housing['total_rooms']

housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing.corr()
pprint.pprint(corr_matrix['median_house_value'].sort_values(ascending=False))

housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

# Data clearning

from sklearn.preprocessing import  Imputer

imputer = Imputer(strategy="median")

housing_num = housing.drop('ocean_proximity', axis=1)

imputer.fit(housing_num)

pprint.pprint(imputer.statistics_)
pprint.pprint(housing_num.median().values)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# Handing text and categorical Attributers
#  way 1

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)
pprint.pprint(housing_cat_encoded)
pprint.pprint(encoder.classes_)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
pprint.pprint(housing_cat_1hot.toarray())

# way 2

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
pprint.pprint(housing_cat_1hot)

# Customer transformers

from sklearn.base import BaseEstimator, TransformerMixin

room_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """Custom transform class."""

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, room_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, room_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# Transformation Pipelines

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # ScandardScaler data
from sklearn.pipeline import FeatureUnion



class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Custom Transformer for sklearn to handle pandas Dataframes."""

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

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

pprint.pprint(housing)
pprint.pprint(housing_prepared.shape)
