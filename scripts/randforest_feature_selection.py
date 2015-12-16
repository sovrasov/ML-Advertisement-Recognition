#!/usr/bin/env python

from __future__ import print_function
from channel_loader import get_small_data
from sklearn.ensemble import RandomForestClassifier
import time, numpy as np, matplotlib.pyplot as plt

def main():

    #channels = ( 'NDTV', 'TIMESNOW', 'CNNIBN', 'CNN', 'BBC' )
    channels = ('NDTV', '')

    n_features = 100

    for channel in channels:
        X, y = get_small_data(channel)
        clf = RandomForestClassifier(n_estimators=60)
        print('Training...')
        t0 = time.clock()
        clf.fit(X, y)
        trainTime = time.clock() - t0

        print('Training time: ' + str(trainTime) + 's\n')
        print('Staritng importances evaluation for {}...'.format(channel))
        #feature_importances_
        #print(clf.feature_importances_)
    
        features = list()
        importances = list()
        for i in range (0, X.shape[1]):
            importances.append(clf.feature_importances_[i])

        importances_sorted = sorted(importances)
        treshold_weight = importances_sorted[n_features]

        mask = list()
        for i in range (0, X.shape[1]):
            mask.append(false)

        for i in range (0, X.shape[1]):
            if clf.feature_importances_[i] >= treshold_weight:
                features.append(i)
                importances.append(clf.feature_importances_[i])
                mask[i] = true

        
        print('Weight {}\n{}'.format(treshold_weight, features))

        plt.clf()
        plt.plot(np.array(features), np.array(importances))
        locs, labels = plt.yticks()
        plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
        plt.ylabel('Feature importance')
        plt.xlabel('Feature')
        plt.savefig('feature-importances-{}.png'.format(channel))
        #print('Evaluation finished')

if __name__ == '__main__':
    main()
