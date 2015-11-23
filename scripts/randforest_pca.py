#!/usr/bin/env python

from __future__ import print_function
from channel_loader import get_data, get_small_data
from PCA_and_CV import do_PCA_and_cv
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition, cross_validation
import time, numpy as np, matplotlib.pyplot as plt

def main():
    X, y = get_small_data('TIMESNOW')

    n_features = list()
    for i in range(1,21):
        n_features.append(i*10)
    n_features.append(228)

    clf = RandomForestClassifier(n_estimators=10)
    t0 = time.clock()
    result = do_PCA_and_cv(clf, X.toarray(), y, n_features)
    testTime = time.clock() - t0
    print('Total time: ' + str(testTime))
    scores = result[0]
    scores_std = result[1]

    plt.clf()
    plt.plot(n_features, scores)
    plt.plot(n_features, np.array(scores) + np.array(scores_std), 'b--')
    plt.plot(n_features, np.array(scores) - np.array(scores_std), 'b--')
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
    plt.ylabel('CV score')
    plt.xlabel('Parameter N')
    plt.show()

if __name__ == '__main__':
    main()
