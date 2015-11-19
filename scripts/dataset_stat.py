#!/usr/bin/env python

from channel_loader import get_data

def get_positives_stat(channel):

    X, y = get_data(channel)
    print('# of ' + channel + ' frames = ' + str(X.shape[0]))
    percent_positives = 100*float(len(filter(lambda x: x == 1, y))) / len(y)

    print('% of positives on ' + channel + ': ' + str(percent_positives))

    return percent_positives

def main():
    avg_positives = 0.0

    avg_positives += get_positives_stat('BBC')
    avg_positives += get_positives_stat('CNN')
    avg_positives += get_positives_stat('CNNIBN')
    avg_positives += get_positives_stat('NDTV')
    avg_positives += get_positives_stat('TIMESNOW')

    print('\nAvg % of positives: ' + str(avg_positives/5.0))

if __name__ == '__main__':
    main()
