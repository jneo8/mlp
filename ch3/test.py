"""Test."""
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from read import read
from confs import logconf
logger = logconf.Logger(__file__).logger


class Test():
    """Tset class."""

    def __init__(self):
        """Init."""
        mnist = read()

        self.x, self.y = mnist['data'], mnist['target']

        # show shape
        logger.info('x.shape {}'.format(self.x.shape))
        logger.info('y.shape {}'.format(self.y.shape))

        # split train&test data set
        self.x_train, self.x_test = self.x[:60000], self.x[60000:]
        self.y_train, self.y_test = self.y[:60000], self.y[60000:]
        shuffle_index = np.random.permutation(60000)
        self.x_train = self.x_train[shuffle_index]
        self.y_train = self.y_train[shuffle_index]
        self.y_test_5 = (self.y_test == 5)
        self.y_train_5 = (self.y_train == 5)
        self.num = 40000  # The target test item between 1 ~ 60000.
        self.some_digit = self.x[self.num]

    @property
    def p1(self):
        """p1."""
        """
        "" Only show one png.
        """

        # reshape data
        some_digit_image = self.some_digit.reshape(28, 28)
        # put digit_image data into plt, and use catimg show in termial.
        plt.imshow(
            some_digit_image,
            cmap=matplotlib.cm.binary,
            interpolation="nearest",
        )
        plt.axis('off')
        plt.savefig('img/c3p1')
        plt.clf()
        subprocess.call(['catimg', '-f', 'img/c3p1.png'])
        logger.info('Label of y[{num}] : {y}'.format(
            num=self.num,
            y=self.y[self.num])
        )

    @property
    def sgd(self):
        """SGD model."""
        # set train num, so that model can guess that is the digit is that num?

        # sklearn.linear_model.SGDClassifier
        self.sgd_clf = SGDClassifier(random_state=42)
        self.sgd_clf.fit(self.x_train, self.y_train_5)
        logger.info('SDG model 5 guess : {}'.format(
            self.sgd_clf.predict([self.some_digit]))
        )

        # Measuring Accuracy using cross-validation
        # 測量精度
        x = cross_val_score(
            self.sgd_clf,
            self.x_train,
            self.y_train_5,
            cv=3,
            scoring="accuracy",
        )

        logger.info('cross_val_score : {}'.format(x))

    @property
    def never5(self):
        """Use nerver5_clf as model."""
        commit = """
        show the performance that always guess 5 :
        """

        never_5_clf = Never5Classifier()
        x = cross_val_score(
            never_5_clf,
            self.x_train,
            self.y_train_5,
            cv=3,
            scoring="accuracy",
        )
        logger.info('{commit}\ncross_val_score use never_5_clf : {x}\n'.format(commit=commit, x=x))

    @property
    def confusion_matrix(self):
        """Confusion martrix."""
        commit = """
        return list [[A, B], [C, D]]
        which mean:

                      guess not 5 | guess 5
        really not 5       A      |   B
        really is  5       C      |   D
        """
        y_train_pred = cross_val_predict(
            self.sgd_clf,
            self.x_train,
            self.y_train_5,
            cv=3,
        )
        x = confusion_matrix(self.y_train_5, y_train_pred)
        logger.info('{commit}\nconfusion_matrix : \n{x}\n'.format(
            commit=commit, x=x
        ))

        ###
        # precision = TP / (TP + FP)
        # which mean TP / (All the items we detect)
        ###
        # sklearn.metrics.precision_score
        presicion = precision_score(self.y_train_5, y_train_pred)
        logger.info('precision {}'.format(presicion))

        ###
        # recall/sensitivity/true positive rate = TP / (TP + FN)
        # which mean TP / (the items should be detect.)
        ###
        # sklearn.metrics.recall_score
        recall = recall_score(self.y_train_5, y_train_pred)
        logger.info('recall {}'.format(recall))

        ###
        # F1-Score = TP / (TP + ((FN + FP) / 2))
        ###
        # sklearn.metrics.f1_score
        f1_score_ = f1_score(self.y_train_5, y_train_pred)
        logger.info('f1 score {}'.format(f1_score_))

        ###
        # Get some_digit's score, and reset threshold.
        ###
        y_score = self.sgd_clf.decision_function([self.some_digit])
        logger.info(
            "some_digit {num} score {score}".format(
                num=self.num,
                score=y_score
            )
        )
        threshold = 20000
        y_some_digit_pred = (y_score > threshold)
        logger.info('y_some_digit_pred : {}'.format(y_some_digit_pred))

    def plot_precsion_recall_vs_threshold(self):
        """Draw curve of presicion, recall and threshold."""
        ###
        # sklearn.metrics.precision_recall_curve
        ###
        y_scores = cross_val_predict(
            self.sgd_clf,
            self.x_train,
            self.y_train_5,
            cv=3,
            method="decision_function",
        )
        precisions, recalls, thresholds = precision_recall_curve(
            self.y_train_5,
            y_scores
        )
        plt.plot(thresholds, precisions[:-1], "b--", label='Precision')
        plt.plot(thresholds, recalls[:-1], "g-", label='Recall')
        plt.xlabel('Threshold')
        plt.legend(loc="upper left")
        plt.ylim([0, 1])

        plt.savefig('img/c3p2')
        plt.clf()
        subprocess.call(['catimg', '-f', 'img/c3p2.png'])


class Never5Classifier(BaseEstimator):
    """Never5Classifier."""

    """
    ""  Return list than pretend the model which  always guess 5.
    """

    def fit(self, x, y=None):
        """Fit data."""
        pass

    def predict(self, x):
        """Predict."""
        return np.zeros((len(x), 1), dtype=bool)


if __name__ == '__main__':
    t = Test()
    t.p1
    t.sgd
    t.never5
    t.confusion_matrix
    t.plot_precsion_recall_vs_threshold()
