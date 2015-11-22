#!/usr/bin/env python

from __future__ import print_function
from channel_loader import get_data, get_small_data
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition, cross_validation
import time, numpy as np, matplotlib.pyplot as plt

def do_PCA_and_cv(clf, X, y, n_features):

    scores = list()
    scores_std = list()
    t0 = time.clock()
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

def main():
    X, y = get_small_data('TIMESNOW')

    n_features = [10, 20, 30, 40, 60, 80]
    clf = RandomForestClassifier(n_estimators=10)
    t0 = time.clock()
    result = do_PCA_and_cv(clf, X.toarray(), y, n_features)
    testTime = time.clock() - t0
    print('Total time: ' + str(testTime))
    scores = result[0]
    scores_std = result[1]

    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.plot(n_features, scores)
    plt.plot(n_features, np.array(scores) + np.array(scores_std), 'b--')
    plt.plot(n_features, np.array(scores) - np.array(scores_std), 'b--')
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
    plt.ylabel('CV score')
    plt.xlabel('Parameter N')
    plt.ylim(0, 1.1)
    plt.show()

if __name__ == '__main__':
    main()
