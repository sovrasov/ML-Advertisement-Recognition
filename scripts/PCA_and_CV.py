#!/usr/bin/env python

from __future__ import print_function
from sklearn import decomposition, cross_validation
import numpy as np
from time import clock, time
from sys import platform
from sklearn.cross_validation import StratifiedKFold

def do_PCA_and_cv(clf, X, y, n_features):

    timer = clock if platform == 'win32' else time
    scores = []
    scores_std = []
    times = []

    for i in range(0, len(n_features)):
        #doPCA
        print('Do PCA with ' + str(n_features[i]) + ' features...')
        pca = decomposition.PCA(n_components = n_features[i])
        #pca.fit(X)
        X_R = pca.fit_transform(X)
        print('PCA done')
        #do CV
        print('Startig cross-validation...')

        current_scores = []
        current_times = []

        for train, test in StratifiedKFold(y, 10):
            X_train, X_test, y_train, y_test = (X_R[train], X_R[test],
                    y[train], y[test])
            start_time = timer()
            clf.fit(X_train, y_train)
            end_time = timer()

            current_scores.append(clf.score(X_test, y_test))
            current_times.append(end_time - start_time)

#        current_scores = cross_validation.cross_val_score(clf, X_R, y, cv=10, n_jobs=-1)
        scores.append(np.mean(current_scores))
        scores_std.append(np.std(current_scores))
        times.append(np.mean(current_times))

        print('Cross-validation done')

    return scores, scores_std, times