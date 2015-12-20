# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np

channels = ['BBC', 'CNN', 'CNNIBN', 'NDTV', 'TIMESNOW']
methods = ['GTB', 'kNN', 'randfor', 'SVM', 'LDA']
label = ''
for method in methods:
   plt.xlabel('Number of features selected')
   plt.ylabel('Mean scores')
   for channel in channels:
	f = open('Files/PCA/PCA-{}-{}.log'.format(method,channel),'r')
	lines = f.readlines()
	f.close()
	#print ('PCA-{}-{}'.format(method,channel))
	string_data = map(lambda x: x.rstrip().split('; '), lines)
	#print(string_data)
	#print len(string_data), len(string_data[0]), len(zip(*string_data)), len(zip(*string_data[0]))
	params, score_means, score_stds, time_means = zip(*string_data)
	#print(zip(*string_data))
	params, score_means, score_stds, time_means = map(float, params),map(float, score_means),map(float, score_stds),map(float, time_means)
	if channel == 'BBC':
	 style = 'b'
	 label = channel
 	elif channel == 'CNN':
	 style = 'c'
	 label = channel
 	elif channel == 'CNNIBN':
	 style = 'r'
	 label = 'CNN-IBN'
 	elif channel == 'NDTV':
	 style = 'g'
	 label = channel
 	elif channel == 'TIMESNOW':
	 style = 'y'
	 label = channel = 'TIMES NOW'
	#line_10 = plt.plot(params, np.array(score_means)-np.array(score_stds), style + '--')
  	#line_20 = plt.plot(params, np.array(score_means)+np.array(score_stds), style + '--')
  	line_30 = plt.plot(params, score_means, style + '-o', label = label)
	plt.grid()
	plt.legend(loc = 'best', fontsize = 10)
	plt.savefig('Pictures/PCA-{}.png'.format(method), format = 'png', dpi = 300)
   plt.clf()
