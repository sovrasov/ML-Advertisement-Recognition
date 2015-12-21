#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys, os

def main():
    if len(sys.argv) < 2:
        print('Base method result directory is not provided. Stopping.')

    path = sys.argv[1]

    methods = ( 'knn', 'lda', 'svm', 'randfor', 'gtb' )
    params = ( '15', None, '4.0', '60', '100' )
    channels = ( 'cnn', 'bbc', 'cnnibn', 'timesnow', 'ndtv' )
    channel_names = ( 'CNN', 'BBC', 'CNN-IBN', 'TIMES NOW', 'NDTV' )

    cell_template = '& \\tworowcell{{\\(Q={:.2f}\\%\\)}}{{\\(T_{{train}}={:.4f}\\) s}} '

    output1 = open('base-table1.txt', 'wt')
    output2 = open('base-table2.txt', 'wt')

    for channel, channel_name in zip(channels, channel_names):
        table_line = '{} '.format(channel_name)

        for method, param in zip(methods, params):
            filename = path + os.sep + '{}-{}.log'.format(method, channel)
            with open(filename, 'rt') as input:
                lines = input.readlines()

            if method == 'lda':
                line = lines[0]
                Q, _, T, _ = line.rstrip().split()
            else:
                line = filter(lambda x: x.startswith(param), lines)[0]
                _, Q, _, T, _ = line.rstrip().split()

            Q, T = float(Q), float(T)
            table_line += cell_template.format(Q * 100.0, T)

            if method == 'svm':
                print(table_line + '\\\\ \\hline', file=output1)
                table_line = '{} '.format(channel_name)
            elif method == 'gtb':
                print(table_line + '\\\\ \\hline', file=output2)

    output1.close()
    output2.close()

            
if __name__ == '__main__':
    main()

