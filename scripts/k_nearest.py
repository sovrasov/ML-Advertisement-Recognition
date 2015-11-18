#!/usr/bin/env python

from channel_loader import get_data
from sklearn import neighbors
import time

def main():
	n_neighbors = 5

	XB, yB = get_data('BBC')
	XC, yC = get_data('CNN')

	print 'Number of BBC frames = ' + str(XB.shape[0]) + '\n'
	print 'Number of CNN frames = ' + str(XC.shape[0])

	clf = neighbors.KNeighborsClassifier(n_neighbors)
	print('Training...')
	t0 = time.clock()
	clf.fit(XB, yB)
	trainTime = time.clock() - t0

	print('Training time: ' + str(trainTime) + 's\n')

	print('Testing...')
	t0 = time.clock()
	score = clf.score(XC[:10000], yC[:10000])
	testTime = time.clock() - t0
	print('Testing time: ' + str(testTime) + 's\n')
	print('Total time: ' + str(trainTime + testTime) + 's\n')
	print 'score = ' + str(score)

if __name__ == '__main__':
    main()