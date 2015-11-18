#!/usr/bin/env python

from channel_loader import get_data
from sklearn import neighbors

n_neighbors = 5

XB, yB = get_data('BBC')
XC, yC = get_data('CNN')

clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(XB, yB)

print 'Number of BBC frames = ' + str(XB.shape[0]) + '\n'
print 'Number of CNN frames = ' + str(XC.shape[0])

score = clf.score(XC[:10000], yC[:10000])
print 'score = ' + str(score)
