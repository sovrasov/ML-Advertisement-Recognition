#!/usr/bin/env python

from __future__ import print_function
from channel_loader import get_data
from sklearn.svm import SVC
import time

def main():
    X_BBC, y_BBC = get_data('BBC')
    X_CNN, y_CNN = get_data('CNN')
    print('# of BBC frames = ' + str(X_BBC.shape[0]))
    print('# of CNN frames = ' + str(X_CNN.shape[0]))

    clf = SVC(C=0.5, cache_size=2000, class_weight='auto', kernel='linear')
    print('Training...')
    t0 = time.clock()
    clf.fit(X_BBC, y_BBC)
    trainTime = time.clock() - t0

    print('Training time: ' + str(trainTime) + 's\n')
    print('Testing...')
    t0 = time.clock()
    score = clf.score(X_CNN, y_CNN)
    testTime = time.clock() - t0

    print('Testing time: ' + str(testTime) + 's\n')
    print('Total time: ' + str(trainTime + testTime) + 's\n')
    print('score = ' + str(score))

if __name__ == '__main__':
    main()
