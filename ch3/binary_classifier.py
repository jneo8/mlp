"""Test."""
import subprocess
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    cross_val_score,
    cross_val_predict,
)
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
from read import read
from confs import logconf
logger = logconf.Logger(__file__).logger


def plot_digits(instances, images_per_row=10, **options):
    """EXTRA."""
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axis("off")


class BinaryClassifier():
    """BinaryClassifier class."""

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
    def main(self):
        """Step by step record."""
        self.guess
        self.sgd
        self.never5
        self.confusion_matrix
        self.plot_precsion_recall_vs_threshold_curve
        self.plot_roc_curve
        self.randforestclassifier_and_roc_curve
        self.multiclass_classification
        self.one_vs_one_classifier
        self.error_analysis
        self.multilabel_classification

    @property
    def guess(self):
        """p1."""
        """
        Only show one png.
        """
        logger.info('Use SGDClassifier to guess img')
        # reshape data
        some_digit_image = self.some_digit.reshape(28, 28)
        # put digit_image data into plt, and use catimg show in termial.
        plt.imshow(
            some_digit_image,
            cmap=matplotlib.cm.binary,
            interpolation="nearest",
        )
        plt.axis('off')
        plt.savefig('img/guess')
        plt.clf()
        subprocess.call(['catimg', '-f', 'img/guess.png'])
        logger.debug('Label of y[{num}] : {y}'.format(
            num=self.num,
            y=self.y[self.num])
        )

    @property
    def sgd(self):
        """SGD model."""
        """
        # set train num, so that model can guess that is the digit is that num?
            sklearn.linear_model.SGDClassifier
        """
        self.sgd_clf = SGDClassifier(random_state=42)
        self.sgd_clf.fit(self.x_train, self.y_train_5)
        logger.debug('SDG model 5 guess : {}'.format(
            self.sgd_clf.predict([self.some_digit]))
        )
        """
        Measuring Accuracy using cross-validation
        測量精度
        """
        x = cross_val_score(
            self.sgd_clf,
            self.x_train,
            self.y_train_5,
            cv=3,
            scoring="accuracy",
        )

        logger.debug('cross_val_score : {}'.format(x))

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
        logger.debug(
            '{commit}\ncross_val_score use never_5_clf : {x}\n'
            .format(commit=commit, x=x)
        )

    @property
    def confusion_matrix(self):
        """Confusion martrix."""
        logger.info('confusion_matrix')
        commit = """
        return list [[A, B], [C, D]]
        which mean:

                      guess not 5 | guess 5
        really not 5       TN      |   FP
        really is  5       FN      |   TP
        """
        y_train_pred = cross_val_predict(
            self.sgd_clf,
            self.x_train,
            self.y_train_5,
            cv=3,
        )
        x = confusion_matrix(self.y_train_5, y_train_pred)
        logger.debug('{commit}\nconfusion_matrix : \n{x}\n'.format(
            commit=commit, x=x
        ))

        """
        precision = TP / (TP + FP)
        which mean TP / (All the items we detect)

        sklearn.metrics.precision_score
        """

        presicion = precision_score(self.y_train_5, y_train_pred)
        logger.debug('precision {}'.format(presicion))

        """
        recall/sensitivity/true positive rate = TP / (TP + FN)
        which mean TP / (the items should be detect.)
        sklearn.metrics.recall_score
        """
        recall = recall_score(self.y_train_5, y_train_pred)
        logger.debug('recall {}'.format(recall))

        """
        F1-Score = TP / (TP + ((FN + FP) / 2))
        """
        # sklearn.metrics.f1_score
        f1_score_ = f1_score(self.y_train_5, y_train_pred)
        logger.debug('f1 score {}'.format(f1_score_))

        """
        Get some_digit's score, and reset threshold.
        """
        y_score = self.sgd_clf.decision_function([self.some_digit])
        threshold = 20000
        y_some_digit_pred = (y_score > threshold)
        logger.debug(
            '\nGet some_digit\'s score, and reset threshold\n'
            "some_digit {num} score {score}\n"
            'y_some_digit_pred, threshold = {threshold}: {y_some_digit_pred}'
            .format(
                num=self.num,
                score=y_score,
                threshold=threshold,
                y_some_digit_pred=y_some_digit_pred,
            )
        )

    @property
    def plot_precsion_recall_vs_threshold_curve(self):
        """Draw curve of presicion, recall and threshold."""
        """
        # Get classifer of 90% precision.
            sklearn.metrics.precision_recall_curve
        """

        self.y_scores = cross_val_predict(
            self.sgd_clf,
            self.x_train,
            self.y_train_5,
            cv=3,
            method="decision_function",
        )

        precisions, recalls, thresholds = precision_recall_curve(
            self.y_train_5,
            self.y_scores,
        )

        # plt
        plt.plot(thresholds, precisions[:-1], "b--", label='Precision')
        plt.plot(thresholds, recalls[:-1], "g-", label='Recall')
        plt.xlabel('Threshold')
        plt.legend(loc="upper left")
        plt.ylim([0, 1])
        plt.savefig('img/precision_recall_curve')
        plt.clf()

        logger.debug('Curve of presicion, recall and threshold')
        subprocess.call(['catimg', '-f', 'img/precision_recall_curve.png'])

        idx_90_pred = 0
        for idx, p in enumerate(precisions):
            if p > 0.9:
                idx_90_pred = idx
                break

        logger.info('Get classifer of 90% precision')

        thresholds_90_pred = thresholds[idx_90_pred]

        y_train_pred_90 = (self.y_scores > thresholds_90_pred)
        logger.debug(
            'precision score when threshold =={threshold} {precision}'
            .format(
                threshold=thresholds_90_pred,
                precision=precision_score(self.y_train_5, y_train_pred_90),
            )
        )

        logger.debug(
            'recall score when threshold == {threshold} {recall}'
            .format(
                threshold=thresholds_90_pred,
                recall=recall_score(self.y_train_5, y_train_pred_90),
            )
        )

    @property
    def plot_roc_curve(self):
        """Plot Roc curve."""
        """
        # plot roc curve.
            sklearn.metrics.roc_curve
        # compute ROC AUC
            sklearn.metrics.roc_auc_score
        # Roc curve
            https://zh.wikipedia.org/wiki/ROC曲线
        """

        logger.info('Plot SGD ROC Curve')
        fpr, tpr, thresholds = roc_curve(self.y_train_5, self.y_scores)

        plt.plot(fpr, tpr, linewidth=2, label=None)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig('img/roc')
        plt.clf()
        subprocess.call(['catimg', '-f', 'img/roc.png'])
        roc_auc = roc_auc_score(self.y_train_5, self.y_scores)
        logger.debug(
            'SGD ROC AUC Score : {score}'
            .format(score=roc_auc)
        )

    @property
    def randforestclassifier_and_roc_curve(self):
        """Plot RandForestClassifier ROC Curve & ROC AUC."""
        """
        #  sklearn.ensemble.Randomforestclassifier
        """
        logger.info('RandomForestClassifier ROC Curve')
        forest_clf = RandomForestClassifier(random_state=42)
        y_probas_forest = cross_val_predict(
            forest_clf,
            self.x_train,
            self.y_train_5,
            cv=3,
            method='predict_proba',
        )
        # score = proba of positive class
        y_score_forest = y_probas_forest[:, 1]

        fpr, tpr, thresholds = roc_curve(self.y_train_5, self.y_scores)
        fpr_forest, tpr_forest, thresholds_forest = roc_curve(
            self.y_train_5,
            y_score_forest,
        )

        # plt
        plt.plot(fpr, tpr, 'b:', label="SGD")
        plt.plot(fpr_forest, tpr_forest, linewidth=2, label="Random Forest")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig('img/randomforestclassifier_roc')
        plt.clf()
        subprocess.call(['catimg', '-f', 'img/randomforestclassifier_roc.png'])

        # ROC AUC
        roc_auc = roc_auc_score(self.y_train_5, self.y_scores)
        randomforestclassifier_roc_auc = roc_auc_score(
            self.y_train_5,
            y_score_forest,
        )
        logger.debug(
            '\nRandomForestClassifier ROC AUC Score : {rfc_score}'
            '\nSGD ROC AUC Score {score}'
            .format(
                rfc_score=randomforestclassifier_roc_auc,
                score=roc_auc,
            )
        )

    @property
    def multiclass_classification(self):
        """Multiclass Classification."""
        logger.info('Multiclass Classification')

        """
        This code trains the SGDClassifier on the trainiing set using the
        origin target classes from 0 to 9 (y_train), instead of the
        5-versus-all target classes.
        Scikit-Learn actually train 10 binary classifiers, got their decision
        scores for the img, and selected the class with the highest score.
        """
        self.sgd_clf.fit(self.x_train, self.y_train)  # y_train, not y_train_5
        logger.debug(self.sgd_clf.predict([self.some_digit]))

        """
        decision_function method return list of scores from all classifier.
        """
        some_digit_scores = self.sgd_clf.decision_function([self.some_digit])
        logger.debug('some_digit_scores {}'.format(some_digit_scores))

        logger.debug('highest score {}'.format(np.argmax(some_digit_scores)))
        logger.debug('sgd_clf.classes_ {}'.format(self.sgd_clf.classes_))
        logger.debug('sgd_clf.classes_[6] {}'.format(self.sgd_clf.classes_[6]))

    @property
    def one_vs_one_classifier(self):
        """User OneVsOneClassifier."""
        """
        # One-versus-one (OvO) strategy
            sklearn.multiclass.OneVsOneClassifier

        # Increase accuracy above
            sklearn.preprocessing.StandardScaler
        """
        logger.info('User OneVsOneClassifier')
        ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
        ovo_clf.fit(self.x_train, self.y_train)
        logger.debug(
            'ovo SGDClassifier Predict some_digiit {}'
            .format(ovo_clf.predict([self.some_digit]))
        )
        logger.debug('len ovo_clf : {}'.format(len(ovo_clf.estimators_)))

        """
        Scikit-learn did not have to run OvO or OvA because Random Forest
        classifier can directly classify instances into multiple classes.
        """
        forest_clf = RandomForestClassifier(random_state=42)
        forest_clf.fit(self.x_train, self.y_train)
        logger.debug(
            'RandForestClassifier  Predict some_digiit {}'
            .format(forest_clf.predict([self.some_digit]))
        )
        logger.debug(
            'RandForestClassifier  Predict proba {}'
            .format(forest_clf.predict_proba([self.some_digit]))
        )

        # evaluate

        ovo_scores = cross_val_score(
            ovo_clf,
            self.x_train,
            self.y_train,
            cv=3,
            scoring='accuracy',
        )
        logger.debug('ovo_scores {}'.format(ovo_scores))

        # Increase accuracy above.
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(self.x_train.astype(np.float64))
        ovo_scaled_scores = cross_val_score(
            ovo_clf,
            x_train_scaled,
            self.y_train,
            cv=3,
            scoring="accuracy",
        )

        logger.debug('ovo_scaled_scores {}'.format(ovo_scaled_scores))

    @property
    def error_analysis(self):
        """Error Analysis."""
        logger.info('Error Analysis')
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(self.x_train.astype(np.float64))
        y_train_pred = cross_val_predict(
            self.sgd_clf,
            x_train_scaled,
            self.y_train,
            cv=3
        )
        conf_mx = confusion_matrix(self.y_train, y_train_pred)
        logger.debug(conf_mx)
        logger.debug(
            '''
            Each rows represent actual classes,
            while columns represent predicted classes.
            '''
        )
        # plt
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.savefig('error_analysis_matrix')
        plt.clf()
        subprocess.call(['catimg', '-f', 'error_analysis_matrix.png'])
        """
        divide each value in the confusion matrix by the number of the image
        in the corresponding class, so you can compare error rates instead of
        absolute number of errors.
        """
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.fill_diagonal.html
        np.fill_diagonal(norm_conf_mx, 0)
        # plt
        plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
        plt.savefig('norm_conf_mx')
        plt.clf()
        subprocess.call(['catimg', '-f', 'norm_conf_mx.png'])

        # plot example of 3s & 5s
        logger.debug('plot example of 3s & 5s')
        cl_a, cl_b = 3, 5
        x_aa = self.x_train[(self.y_train == cl_a) & (y_train_pred == cl_a)]
        x_ab = self.x_train[(self.y_train == cl_a) & (y_train_pred == cl_b)]
        x_ba = self.x_train[(self.y_train == cl_b) & (y_train_pred == cl_a)]
        x_bb = self.x_train[(self.y_train == cl_b) & (y_train_pred == cl_b)]
        plt.figure(figsize=(8, 8))
        plt.subplot(221)
        plot_digits(x_aa[:25], images_per_row=5)
        plt.subplot(222)
        plot_digits(x_ab[:25], images_per_row=5)
        plt.subplot(223)
        plot_digits(x_ba[:25], images_per_row=5)
        plt.subplot(224)
        plot_digits(x_bb[:25], images_per_row=5)
        plt.savefig('plot_example_3s&5s')
        plt.clf()
        subprocess.call(['catimg', '-f', 'plot_example_3s&5s.png'])

    @property
    def multilabel_classification(self):
        """mulitilabel_classification."""
        """
        sklearn.neighbors.KNeighborsClassifier
        Create y_multilabel array containing two target labels
        for each digit img.
        """
        y_train_large = (self.y_train >= 7)
        y_train_odd = (self.y_train % 2 == 1)
        y_multilabel = np.c_[y_train_large, y_train_odd]

        knn_clf = KNeighborsClassifier()
        knn_clf.fit(self.x_train, y_multilabel)

        logger.debug(
            'Knn predict {value}'
            .format(
                value=knn_clf.predict([self.some_digit])
            )
        )


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
    starttime = time.time()
    t = BinaryClassifier()
    t.main
    endtime = time.time()
    logger.info('total time {}'.format(endtime - starttime))



