"""For every record calculate a statistic for classifying the record

The result is written both as a pickle file and a text file with lines
like

c04 N 0.0399
x29 N 0.0471
x11 N 0.1840

"""
import sys
import os
import argparse
import glob
import pickle
import typing

import numpy
import pint

import hmmds.applications.apnea.utilities

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser("Create and write/pickle pass1_report")
    parser.add_argument('--sample_rate_in',
                        type=int,
                        default=2,
                        help='Samples per second of input')
    parser.add_argument('--data_dir',
                        type=str,
                        default='../../../../build/derived_data/ECG/',
                        help='Path to heart rate data for reading')
    parser.add_argument(
        '--border',
        type=float,
        default=0.36,
        help='Border between normal and apnea for whole records')
    parser.add_argument('--format',
                        type=str,
                        default='{0}/{1}_self_AR3/heart_rate',
                        help='Map from (data_dir,name) to file of heart_rates')
    parser.add_argument('pickle', type=str, help='Path to pickled result')
    hmmds.applications.apnea.utilities.common_arguments(parser)
    args = parser.parse_args(argv)
    args.sample_rate_in *= PINT('Hz')
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    records = dict((name, hmmds.applications.apnea.utilities.Pass1(name, args))
                   for name in args.all_names)
    all_names = args.all_names.copy()
    all_names.sort(key=lambda x: records[x].statistic_1())
    result = {}
    for name in all_names:
        statistic = records[name].statistic_1()
        _class = ("N", "A")[int(statistic > args.border)]
        print(f'{name} {_class} {statistic:6.4f}')
        result[name] = _class
    with open(args.pickle, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
