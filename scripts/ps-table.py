#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys, os

def main():
    if len(sys.argv) < 4:
        print('Not enough arguments.')
        sys.exit(1)

    base_path = sys.argv[1]
    red_path = sys.argv[2]
    prefix = sys.argv[3]

    methods = ( 'knn', 'lda', 'svm', 'randfor', 'gtb' )
    red_methods = ( 'KNN', 'LDA', 'SVM', 'randforest', 'gradboosting' )
    params = ( '15', None, '4.0', '60', '100' )
    channels = ( 'cnn', 'bbc', 'cnnibn', 'timesnow', 'ndtv' )
    channel_names = ( 'CNN', 'BBC', 'CNN-IBN', 'TIMES NOW', 'NDTV' )

    stats_cell_template = '& \\tworowcell{{\(R={:.2f}\\%\\)}}{{\\(T_{{PS}}={:.2f}\\) s}} ' if prefix != 'ccis' else '& \\(R={:.2f}\\%\\)'
    cell_template = '& \\tworowcell{{\\(Q={:.2f}\\%\\;({:+.2f}\\%)\\)}}{{\\(T_{{tr}}={:.3f}\\) s \\((\mathrm{{x}}\;{:.3f})\\)}} '

    output1 = open('{}-table1.txt'.format(prefix), 'wt')
    output2 = open('{}-table2.txt'.format(prefix), 'wt')

    for channel, channel_name in zip(channels, channel_names):
        table_line = '{} '.format(channel_name)

        stats_path = '{0}/../{1}-stats/{1}-{2}-stats.log' if prefix != 'ccis' else \
                '{0}/{1}-{2}-stats.log'
        with open(stats_path.format(red_path, prefix, channel.upper()),
                'rt') as input:
            _, R, _, Tps, _ = map(float, input.readline().rstrip().split())
        table_line += stats_cell_template.format(R * 100.0, Tps)

        for method, red_method, param in zip(methods, red_methods, params):
            filename = base_path + os.sep + '{}-{}.log'.format(method, channel)
            with open(filename, 'rt') as input:
                lines = input.readlines()

            if method == 'lda':
                line = lines[0]
                Q, _, T, _ = line.rstrip().split()
            else:
                line = filter(lambda x: x.startswith(param), lines)[0]
                _, Q, _, T, _ = line.rstrip().split()

            Q, T = float(Q), float(T)

            filename = red_path + os.sep + '{}-{}-{}.log'.format(prefix,
                    red_method, channel.upper())
            with open(filename, 'rt') as input:
                _, red_Q, _, red_T, _ = map(float,
                        input.readline().rstrip().split())

            table_line += cell_template.format(red_Q * 100.0,
                    (red_Q - Q) * 100.0, red_T, red_T / T)

            if method == 'lda':
                print(table_line + '\\\\ \\hline', file=output1)
                table_line = '{} '.format(channel_name)
            elif method == 'gtb':
                print(table_line + '\\\\ \\hline', file=output2)

    output1.close()
    output2.close()

            
if __name__ == '__main__':
    main()

