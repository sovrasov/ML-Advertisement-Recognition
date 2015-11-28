#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn
if sklearn.__version__  < '0.17':
    from sklearn.lda import LDA
else:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold

from channel_loader import get_data, get_small_data

class OptimizerBase(object):
    def __init__(self, param_name, start, stop, step):
        self._start = start
        self._stop = stop
        self._step = step
        self._mean_scores = []
        self._score_stds = []
        self._mean_t_train = []
        self._t_train_stds = []
        self._param_name = param_name
        self._param_grid = np.arange(start, stop, step)
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
                t0 = self._timer()
                clf.fit(X_train.toarray(), y_train)
                current_train_times.append(self._timer() - t0)
                current_param_scores.append(clf.score(X_test.toarray(), y_test))

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
        with open('base-methods.log', 'at') as logfile:
            print('{} {}:'.format(self._method_name, channel_name), file=logfile)
            for data_item in data:
                print('\tparam = {}, Q = {}±{}, T_train = {} ± {}'.format(
                    *data_item), file=logfile)


class KNNOptimizer(OptimizerBase):
    def __init__(self):
        super(KNNOptimizer, self).__init__('n_neighbors', 1, 102, 2)
        self._clfclass = KNeighborsClassifier
        self._method_name = 'kNN'


class RandForestOptimizer(OptimizerBase):
    def __init__(self):
        super(RandForestOptimizer, self).__init__('n_estimators', 5, 106, 5)
        self._clfclass = RandomForestClassifier
        self._method_name = 'Random forest'


class GTBOptimizer(OptimizerBase):
    def __init__(self):
        super(GTBOptimizer, self).__init__('n_estimators', 10, 301, 10)
        self._clfclass = GradientBoostingClassifier
        self._method_name = 'GTB'


class LDAOptimizer(object):
    def __init__(self):
        self._timer = time.clock if sys.platform == 'win32' else time.time
        self._method_name = 'LDA'

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
            train_times.append(t0 - self._timer())

        self._mean_score = np.mean(scores)
        self._score_std = np.var(scores)
        self._mean_train_time = np.mean(train_times)
        self._train_time_std = np.var(train_times)

    def plot_results(self, channel):
        pass

    def log_results(self, channel_name):
        with open('base-methods.log', 'at') as logfile:
            print('{} {}:'.format(self._method_name, channel_name), file=logfile)
            print('\tQ = {}±{}, T_train = {} ± {}'.format(
                self._mean_score, self._score_std,
                self._mean_train_time, self._train_time_std), file=logfile)


def main():
    optimizers = ( KNNOptimizer, LDAOptimizer, # SVMOptimizer,
            RandForestOptimizer, GTBOptimizer )
    channels = ( 'CNN', 'BBC', 'CNNIBN', 'TIMESNOW', 'NDTV' )
    for channel in channels:
        X, y = get_small_data(channel)
        for opt in optimizers:
            optimizer = opt()
            optimizer.optimize(X, y)
            optimizer.log_results(channel)
            optimizer.plot_results(channel)

if __name__ == '__main__':
    main()
