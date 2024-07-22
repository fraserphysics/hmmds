"""For every record calculate a statistic for classifying the record

python pass1.py ../../../../build/derived_data/apnea/statistics2048.pkl pass1.out

The result is written both as a pickle file and a text file with lines
like

c07 N -0.0900
x24 N 0.0842
b04 N 0.1288
x03 A 0.1919
b01 A 0.4121
a07 A 0.4406

"""
import sys
import argparse
import pickle
import typing

import numpy

import hmmds.applications.apnea.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser("Create and write/pickle pass1_report")
    parser.add_argument(
        '--border',
        type=float,
        default=0.175,
        help='Border between normal and apnea for whole records')
    parser.add_argument('statistics',
                        type=argparse.FileType('rb'),
                        help='Path to statistics for pass1')
    parser.add_argument('pickle', type=str, help='Path to pickled result')
    hmmds.applications.apnea.utilities.common_arguments(parser)
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    statistics = pickle.load(args.statistics)
    coefficients = statistics['pass1_coefficients'].reshape(-1)
    psds = statistics['psds']
    statistic_1 = dict((name, numpy.dot(coefficients, numpy.log10(psds[name])))
                       for name in args.all_names)
    all_names = args.all_names.copy()
    all_names.sort(key=lambda x: statistic_1[x])

    result = {}
    for name in all_names:
        statistic = statistic_1[name]
        _class = ("N", "A")[int(statistic > args.border)]
        print(f'{name} {_class} {statistic:6.4f}')
        result[name] = _class
    with open(args.pickle, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
