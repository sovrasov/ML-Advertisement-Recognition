#!/usr/bin/env python

from __future__ import print_function
from sklearn import decomposition, cross_validation
import numpy as np

def do_PCA_and_cv(clf, X, y, n_features):

    scores = list()
    scores_std = list()
    for i in range(0, len(n_features)):
        #doPCA
        print('Do PCA with ' + str(n_features[i]) + ' features...')
        pca = decomposition.PCA(n_components = n_features[i])
        #pca.fit(X)
        X_R = pca.fit_transform(X)
        print('PCA done')
        #do CV
        print('Startig cross-validation...')
        current_scores = cross_validation.cross_val_score(clf, X_R, y, cv=10, n_jobs=-1)
        scores.append(np.mean(current_scores))
        scores_std.append(np.std(current_scores))
        print('Cross-validation done')

    return scores, scores_std