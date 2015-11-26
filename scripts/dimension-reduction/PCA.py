#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from time import clock, time
from sys import platform
from sys import argv
from copy import deepcopy

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import sklearn
if sklearn.__version__ < '0.17':
    from sklearn.lda import LDA
else:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold


def main():
	f = open('PCA results.txt', 'w')

	channels = ( 'NDTV', 'TIMESNOW', 'CNNIBN', 'CNN', 'BBC' )
	learn_methods = ({ 'class': KNeighborsClassifier, 'name': 'kNN',
	'params': {'n_neighbors': 5}, 'dense_X': False },
	{ 'class': LDA, 'name': 'LDA', 'params': {}, 'dense_X': True },
	{ 'class': SVC, 'name': 'SVM', 'params': {}, 'dense_X': False },
	{ 'class': RandomForestClassifier, 'name': 'Random forest',
	'params': {'n_estimators': 50}, 'dense_X': False },
	{ 'class': GradientBoostingClassifier,
	'name': 'Gradient tree boosting',
	'params': {'n_estimators': 100}, 'dense_X': True })

	pca_n_features = []
	for i in range(1,21):
	    pca_n_features.append(i*10)
	pca_n_features.append(227)

	timer = clock if platform == 'win32' else time

	for channel in channels:
	    XBig, y = load_svmlight_file('../../Dataset/{}.txt'.format(channel))
	    XBig = VarianceThreshold().fit_transform(XBig)
	    print('Loaded {} dataset...'.format(channel))
	    reduction_rates = []
	    ps_times = []
	    scores = { 'kNN': [], 'LDA': [], 'SVM': [], 'Random forest': [],
	            'Gradient tree boosting': [] }
	    train_times = deepcopy(scores)

	    Xs_reduced = list()

	    print('Starting PCA...')
	    for n_features in pca_n_features:
	    	start_time = timer()
	    	pca = PCA(n_components = n_features)
	    	Xs_reduced.append(pca.fit_transform(XBig.toarray()))
	    	end_time = timer()
	    	ps_times.append(end_time - start_time)
	    print('PCA finished')

	    for X in Xs_reduced:
	    	print('Model dimension = {}'.format(X[0].shape[0]))
	    	f.write('Model dimension = {}'.format(X[0].shape[0]))
	        for train, test in StratifiedKFold(y, 10):
	            X_train, X_test, y_train, y_test = (X[train], X[test],
	                    y[train], y[test])
	            #X_train = X_train.toarray()
	            #X_test = X_test.toarray()
	            # print('Selecting prototypes...')
	            #start_time = timer()
	            #ps = PrototypeSelector(X_train, y_train.astype(np.int))
	            #X_train_red, y_train_red = ps.fcnn_reduce(int(argv[1]))
	            #end_time = timer()
	            #ps_times.append(end_time - start_time)
	            #reduction_rates.append(X_train_red.shape[0] / X_train.shape[0])
	            # print('{}% of {} instances selected in {} s.'.format(
	            # 100 * reduction_rates[-1], X_train.shape[0], ps_times[-1]))

	            for method in learn_methods:
	                method_class = method['class']
	                method_params = method['params']
	                print('Testing with {}...'.format(method['name']))

	                clf = method['class'](**method['params'])
	                #print('Training...')
	                start_time = timer()
	                clf.fit(X_train, y_train)
	                end_time = timer()
	                train_times[method['name']].append(end_time - start_time)

	                #print('Testing...')
	                scores[method['name']].append(clf.score(X_test, y_test))

	        for method in learn_methods:
	            mean_score = np.mean(scores[method['name']])
	            score_variance = np.var(scores[method['name']])
	            mean_train_time = np.mean(train_times[method['name']])
	            train_time_variance = np.var(train_times[method['name']])

	            print('{}, {}: Q = {}±{}, Ttr = {}±{}'.format(
	                channel, method['name'], mean_score, score_variance,
	                mean_train_time, train_time_variance))

	            f.write('{}, {}: Q = {}±{}, Ttr = {}±{}'.format(
	                channel, method['name'], mean_score, score_variance,
	                mean_train_time, train_time_variance))
	f.close()
"""
	        mean_reduction_rate = np.mean(reduction_rates)
	        reduction_rate_variance = np.var(reduction_rates)
	        mean_ps_time = np.mean(ps_times)
	        ps_time_variance = np.var(ps_times)

	        print('R = {}±{}, Tps = {}±{}'.format(mean_reduction_rate,
	            reduction_rate_variance, mean_ps_time, ps_time_variance))
"""
if __name__ == '__main__':
	main()