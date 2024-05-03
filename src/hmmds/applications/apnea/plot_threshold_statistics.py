"""threshold_statistics.py Explore dependence of threshold on statistics

python threshold_statistics.py threshold_statistics.pkl plot.pdf

Derived from shift_threshold.py
"""
from __future__ import annotations

import sys
import argparse
import typing
import pickle

import numpy

import utilities
import plotscripts.utilities
from threshold_statistics import Fit


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Plot of statistics for predicting best threshold")
    utilities.common_arguments(parser)
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('parameter_path', type=str, help="path to pickled data")
    parser.add_argument('result_path', type=str, help="path to pdf file")
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def print_thresholds(statistics):
    """Print list of records and best thresholds sorted by threshold
    """
    names = list(statistics.keys())
    names.sort(key=lambda x: statistics[x][5])
    for name in names:
        print(f'{name} {statistics[name][5]}')


def main(argv=None):
    """Plot best thresholds on training data against various statistics

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    with open(args.parameter_path, 'rb') as _file:
        record_names, fit, statistics = pickle.load(_file)

    # print_thresholds(statistics)
    fig, (pass1_axes, like_a_axes, like_c_axes,
          fit_axes) = pyplot.subplots(nrows=4, ncols=1, figsize=(8, 12))

    for name in record_names:  #fit_threshold in zip(args.record_names, fit_thresholds):
        x = statistics[name][0]  # pass1
        y = statistics[name][5]  # Best threshold
        pass1_axes.semilogy(x,
                            y,
                            marker=f'${name}$',
                            markersize=14,
                            linestyle='None')
        pass1_axes.set_xlabel('pass1')

        x = statistics[name][1]
        like_a_axes.semilogy(x,
                             y,
                             marker=f'${name}$',
                             markersize=14,
                             linestyle='None')
        like_a_axes.set_xlabel('like_a')

        x = statistics[name][2]
        like_c_axes.semilogy(x,
                             y,
                             marker=f'${name}$',
                             markersize=14,
                             linestyle='None')
        like_c_axes.set_xlabel('-like_c')

        x = statistics[name][4]  # Fit threshold
        fit_axes.loglog(x,
                        y,
                        marker=f'${name}$',
                        markersize=14,
                        linestyle='None')
        fit_axes.loglog([1e-3, 1e4], [1e-3, 1e4])
        fit_axes.set_xlabel('fit')

    fig.tight_layout()
    fig.savefig(args.result_path)

    if args.show:
        pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
