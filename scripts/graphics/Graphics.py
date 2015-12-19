# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np

channels = ['bbc', 'cnn', 'cnnibn', 'ndtv', 'timesnow']
methods = ['gtb', 'knn', 'randfor', 'svm']
label = ''
for method in methods:
   plt.ylabel('Mean scores')
   if method == 'gtb':
	plt.xlabel('Number of trees')
   elif method == 'knn':
	plt.xlabel('Number of neighbours')
   elif method == 'randfor':
	plt.xlabel('Number of trees')
   elif method == 'svm':
	plt.xlabel('Parameter C'), plt.xscale('log',basex = 2)
   for channel in channels:
	f = open('Files/{}-{}.log'.format(method,channel),'r')
	lines = f.readlines()
	f.close()
	string_data = map(lambda x: x.rstrip().split(), lines)
	params, score_means, score_stds, time_means, time_stds = zip(*string_data)
	params, score_means, score_stds, time_means, time_stds = map(float, params),map(float, score_means),map(float, score_stds),map(float, time_means),map(float, time_stds)
	if channel == 'bbc':
	 style = 'b'
	 label = 'BBC'
 	elif channel == 'cnn':
	 style = 'c'
	 label = 'CNN'
 	elif channel == 'cnnibn':
	 style = 'r'
	 label = 'CNN-IBN'
 	elif channel == 'ndtv':
	 style = 'g'
	 label = 'NDTV'
 	elif channel == 'timesnow':
	 style = 'y'
	 label = 'TIMES NOW'
  	#line_10 = plt.plot(params, np.array(score_means)-np.array(score_stds), style + '--')
  	#line_20 = plt.plot(params, np.array(score_means)+np.array(score_stds), style + '--')
  	line_30 = plt.plot(params, score_means, style + '-o', label = label)
	plt.grid()
	plt.legend(loc = 'best', fontsize = 10)
	plt.savefig('Pictures/{}.png'.format(method), format = 'png', dpi = 300)
   plt.clf()
