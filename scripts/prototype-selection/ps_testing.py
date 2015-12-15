#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from time import clock, time
from sys import platform, argv
from copy import deepcopy

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sklearn
if sklearn.__version__ < '0.17':
    from sklearn.lda import LDA
else:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from prototype_selection import PrototypeSelector as PS


def main():
    ps_method = argv[1]
    channels = ( 'NDTV', 'TIMESNOW', 'CNNIBN', 'CNN', 'BBC' )
    learn_methods = (
            { 'clf': KNeighborsClassifier(n_neighbors=15), 'name': 'KNN' },
            { 'clf': LDA(), 'name': 'LDA' },
            { 'clf': Pipeline([('scaler', StandardScaler()),
                ('svm', SVC(C=4.0))]), 'name': 'SVM' },
            { 'clf': RandomForestClassifier(n_estimators=60),
                'name': 'randforest' },
            { 'clf': GradientBoostingClassifier(n_estimators=100),
                'name': 'gradboosting' }
    )

    timer = clock if platform == 'win32' else time
    ps_method_function = PS.cnn_reduce if ps_method == 'cnn' else PS.fcnn_reduce

    for ps_param in (1, 5, 10, 15):
        for channel in channels:
            X, y = load_svmlight_file('../../Dataset/{}.txt'.format(channel))
            X = VarianceThreshold().fit_transform(X)
            print('Loaded {} dataset...'.format(channel))
            reduction_rates = []
            ps_times = []
            scores = { 'KNN': [], 'LDA': [], 'SVM': [], 'randforest': [],
                    'gradboosting': [] }
            train_times = deepcopy(scores)

            for train, test in StratifiedKFold(y, 10):
                X_train, X_test, y_train, y_test = (X[train], X[test],
                        y[train], y[test])
                X_train = X_train.toarray()
                X_test = X_test.toarray()
                start_time = timer()
                ps = PS(X_train, y_train.astype(np.int))
                X_train_red, y_train_red = ps_method_function(ps, ps_param)
                end_time = timer()
                ps_times.append(end_time - start_time)
                reduction_rates.append(X_train_red.shape[0] / X_train.shape[0])

                for method in learn_methods:
                    print('Testing with {}...'.format(method['name']))

                    clf = method['clf']
                    start_time = timer()
                    clf.fit(X_train_red, y_train_red)
                    end_time = timer()
                    train_times[method['name']].append(end_time - start_time)

                    scores[method['name']].append(clf.score(X_test, y_test))

            for method in learn_methods:
                mean_score = np.mean(scores[method['name']])
                score_variance = np.var(scores[method['name']])
                mean_train_time = np.mean(train_times[method['name']])
                train_time_variance = np.var(train_times[method['name']])

                with open('{}-{}-{}.log'.format(ps_method,
                    method['name'], channel), 'at') as logfile:
                    print('{} {} {} {} {}'.format(
                        ps_param, mean_score, score_variance,
                        mean_train_time, train_time_variance), file=logfile)

            mean_reduction_rate = np.mean(reduction_rates)
            reduction_rate_variance = np.var(reduction_rates)
            mean_ps_time = np.mean(ps_times)
            ps_time_variance = np.var(ps_times)

            with open('{}-{}-stats.log'.format(ps_method,
                channel), 'at') as logfile:
                print('{} {} {} {} {}'.format(ps_param, mean_reduction_rate,
                    reduction_rate_variance, mean_ps_time, ps_time_variance),
                    file=logfile)


if __name__ == '__main__':
    main()

