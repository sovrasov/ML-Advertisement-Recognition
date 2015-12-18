#!/usr/bin/env python

from __future__ import print_function
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from time import clock, time
from sys import platform
from sklearn.cross_validation import StratifiedKFold


def do_randfor_selection(clf, X, y, n_features):

    timer = clock if platform == 'win32' else time
    scores = []
    scores_std = []
    times = []

    for i in range(0, len(n_features)):
        print('Do selection with ' + str(n_features[i]) + ' features...')

        selector = RandomForestClassifier(n_estimators=40, n_jobs = 1)
        selector.fit(X, y)

        features = list()
        importances = list()

        for j in range (0, X.shape[1]):
            importances.append(selector.feature_importances_[j])

        importances_sorted = sorted(importances)
        treshold_weight = importances_sorted[n_features[i] - 1]

        mask = list()
        for j in range (0, X.shape[1]):
            mask.append(False)
        mask = np.array(mask)

        for j in range (0, X.shape[1]):
            if importances[i] > treshold_weight and len(features) <= n_features[i]:
                features.append(j)
                mask[j] = True
                
        Xt = np.transpose(X)
        Xt = Xt[mask]
        X_R = np.transpose(Xt)

        print('Selection done')
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

        print('Cross-validation done with score {}, time {}'.format(np.mean(current_scores), np.mean(current_times)))

    return scores, scores_std, times