# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np

channels = ['BBC', 'CNN', 'CNNIBN', 'NDTV', 'TIMESNOW']
methods = ['GTB', 'kNN', 'randfor', 'SVM', 'LDA']
label = ''
for method in methods:
   plt.xlabel('Number of features selected')
   plt.ylabel('Mean learning times, s')
   for channel in channels:
	f = open('Files/PCA/PCA-{}-{}.log'.format(method,channel),'r')
	lines = f.readlines()
	f.close()
	string_data = map(lambda x: x.rstrip().split('; '), lines)
	params, score_means, score_stds, time_means = zip(*string_data)
	params, score_means, score_stds, time_means = map(float, params),map(float, score_means),map(float, score_stds),map(float, time_means)
	if channel == 'BBC':
	 style = 'b'
	 label = 'BBC'
 	elif channel == 'CNN':
	 style = 'c'
	 label = "CNN"
 	elif channel == 'CNN-IBN':
	 style = 'r'
	 label = 'CNN-IBN'
 	elif channel == 'NDTV':
	 style = 'g'
	 label = 'NDTV'
 	elif channel == 'TIMESNOW':
	 style = 'y'
	 label = 'TIMES NOW'
	line_10 = plt.plot(params, np.array(time_means), style + '-o', label = channel)
   plt.grid()
   plt.legend(loc = 'best', fontsize = 10)
   plt.savefig('Pictures/PCA-{}Time.png'.format(method), format = 'png', dpi = 300)
   plt.clf()
