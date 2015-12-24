# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np

channels = ['BBC', 'CNN', 'CNNIBN', 'NDTV', 'TIMESNOW']
methods = ['gradboosting', 'KNN', 'randforest', 'SVM', 'LDA']

for method in methods:
   plt.ylabel('Mean learning times, s')
   plt.xlabel('Number of neighbors')
   for channel in channels:
	f = open('cnn/cnn-{}-{}.log'.format(method,channel),'r')
	lines = f.readlines()
	f.close()
	string_data = map(lambda x: x.rstrip().split(), lines)
        string_data = string_data[:3]
	params, score_means, score_stds, time_means, time_stds = zip(*string_data)
	params, score_means, score_stds, time_means, time_stds = map(float, params),map(float, score_means),map(float, score_stds),map(float, time_means),map(float, time_stds)
	if channel == 'BBC':
	 style = 'b'
         label = 'BBC'
 	elif channel == 'CNN':
	 style = 'c'
         label = 'CNN'
 	elif channel == 'CNNIBN':
	 style = 'r'
         label = 'CNN-IBN'
 	elif channel == 'NDTV':
	 style = 'g'
         label = 'NDTV'
 	elif channel == 'TIMESNOW':
	 style = 'y'
         label = 'TIMES NOW'
	line_10 = plt.plot(params, np.array(time_means), style + '-o', label=label)
   plt.grid()
   plt.legend(loc = 'best', fontsize = 10)
   plt.savefig('Pictures/cnn-{}Time.png'.format(method), format = 'png', dpi = 300)
   plt.clf()
