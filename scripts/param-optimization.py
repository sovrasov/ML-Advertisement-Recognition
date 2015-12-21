#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.neighbors import KNeighborsClassifier
import sklearn
if sklearn.__version__  < '0.17':
    from sklearn.lda import LDA
else:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing

from channel_loader import get_data, get_small_data

class OptimizerBase(object):
    def __init__(self, param_name, start, stop, step, scale='lin'):
        self._start = start
        self._stop = stop
        self._step = step
        self._mean_scores = []
        self._score_stds = []
        self._mean_t_train = []
        self._t_train_stds = []
        self._param_name = param_name
        if scale == 'lin':
            self._param_grid = np.arange(start, stop, step)
        elif scale == 'log':
            self._param_grid = np.logspace(start, stop,
                    abs(stop - start) + 1, base=step)
        self._timer = time.clock if sys.platform == 'win32' else time.time
        self._clfclass = None

    def optimize(self, X, y):
        if self._clfclass == None:
            raise ValueError('Empty classifier')
        for param in self._param_grid:
            clf = self._clfclass(**{ self._param_name: param })
            current_param_scores = []
            current_train_times = []
            for train, test in StratifiedKFold(y, 10):
                X_train, X_test, y_train, y_test = (X[train], X[test],
                        y[train], y[test])
                X_train = X_train.toarray() if issparse(X_train) else X_train
                X_test = X_test.toarray() if issparse(X_test) else X_test
                t0 = self._timer()
                clf.fit(X_train, y_train)
                current_train_times.append(self._timer() - t0)
                current_param_scores.append(clf.score(X_test, y_test))

            self._mean_scores.append(np.mean(current_param_scores))
            self._score_stds.append(np.var(current_param_scores))
            self._mean_t_train.append(np.mean(current_train_times))
            self._t_train_stds.append(np.var(current_train_times))

    def plot_results(self, channel_name):
        plt.clf()
        plt.plot(self._param_grid, self._mean_scores, 'r-o')
        plt.plot(self._param_grid,
                np.array(self._mean_scores) - np.array(self._score_stds),
                'r--')
        plt.plot(self._param_grid,
                np.array(self._mean_scores) + np.array(self._score_stds),
                'r--')
        plt.grid()
        plt.xlabel(self._param_name)
        plt.ylabel('{} score'.format(self._method_name))
        plt.savefig('{}-{}.png'.format(self._method_name, channel_name),
                dpi=200)

    def log_results(self, channel_name):
        data = zip(self._param_grid, self._mean_scores, self._score_stds,
                self._mean_t_train, self._t_train_stds)
        path = '{}-{}.log'.format(self._method_name, channel_name)
        with open(path, 'at') as logfile:
            # print('{} {} {}'.format(self._method_name, channel_name,
                # len(self._param_grid)), file=logfile)
            for data_item in data:
                print('{} {} {} {} {}'.format(
                    *data_item), file=logfile)


class KNNOptimizer(OptimizerBase):
    def __init__(self):
        super(KNNOptimizer, self).__init__('n_neighbors', 1, 102, 2)
        self._clfclass = KNeighborsClassifier
        self._method_name = 'knn'


class RandForestOptimizer(OptimizerBase):
    def __init__(self):
        super(RandForestOptimizer, self).__init__('n_estimators', 5, 126, 5)
        self._clfclass = RandomForestClassifier
        self._method_name = 'randfor'

class SVMOptimizer(OptimizerBase):
    def __init__(self):
        super(SVMOptimizer, self).__init__('C', -1, 5, 2.0, scale='log')
        self._clfclass = SVC
        self._method_name = 'svm'

    def optimize(self, X, y):
        X_scaled = preprocessing.scale(X.toarray())
        super(self.__class__, self).optimize(X_scaled, y)


class GTBOptimizer(OptimizerBase):
    def __init__(self):
        super(GTBOptimizer, self).__init__('n_estimators', 50, 171, 10)
        self._clfclass = GradientBoostingClassifier
        self._method_name = 'gtb'


class LDAOptimizer(object):
    def __init__(self):
        self._timer = time.clock if sys.platform == 'win32' else time.time
        self._method_name = 'lda'

    def optimize(self, X, y):
        clf = LDA()
        scores = []
        train_times = []
        for train, test in StratifiedKFold(y, 10):
            X_train, X_test, y_train, y_test = (X[train], X[test],
                    y[train], y[test])
            clf.fit(X_train.toarray(), y_train)
            t0 = self._timer()
            scores.append(clf.score(X_test.toarray(), y_test))
            train_times.append(self._timer() - t0)

        self._mean_score = np.mean(scores)
        self._score_std = np.var(scores)
        self._mean_train_time = np.mean(train_times)
        self._train_time_std = np.var(train_times)

    def plot_results(self, channel):
        pass

    def log_results(self, channel_name):
        path = '{}-{}.log'.format(self._method_name, channel_name)
        with open(path, 'wt') as logfile:
            # print('{} {} {}'.format(self._method_name, channel_name,
                # len(self._param_grid)), file=logfile)
            print('{} {} {} {}'.format(
                self._mean_score, self._score_std,
                self._mean_train_time, self._train_time_std), file=logfile)


def main():
    if len(sys.argv) < 3:
        print('Invalid params. USAGE: param_optimization.py <method> <channel>')
        print('Available methods:\n\tknn, lda, svm, randfor, gtb')
        print('Available channels:\n\tcnn, bbc, ndtv, timesnow, cnnibn')
        sys.exit()

    method_name, channel_name = sys.argv[1], sys.argv[2]

    optimizers = { 'knn': KNNOptimizer, 'lda': LDAOptimizer, 'svm': SVMOptimizer,
            'randfor': RandForestOptimizer, 'gtb': GTBOptimizer }
    X, y = get_small_data(channel_name.upper())
    optimizer = optimizers[method_name]()
    optimizer.optimize(X, y)
    optimizer.log_results(channel_name)
    # optimizer.plot_results(channel_name)

if __name__ == '__main__':
    main()
