"""threshold_statistics.py Explore dependence of threshold on statistics

python threshold_statistics.py default threshold_statistics.pdf a01 a02 ...

Derived from shift_threshold.py
"""
from __future__ import annotations

import sys
import argparse
import typing
import pickle

import numpy
import scipy.optimize

import utilities
import plotscripts.utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Plot to check if statistics predict best threshold")
    utilities.common_arguments(parser)
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('a_model_path', type=str, help="path to model")
    parser.add_argument('record_names', type=str, nargs='+')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def minimum_error(model_record):
    """Find rough approximation of threshold that minimizes error for
    a record.

    """
    # 10 -> 1000 decreases the number of errors 1,444 -> 1,363
    thresholds = numpy.geomspace(.001, 250, 10)
    objective_values = numpy.zeros(len(thresholds), dtype=int)
    counts = numpy.empty((len(thresholds), 4), dtype=int)
    for i, threshold in enumerate(thresholds):
        model_record.classify(threshold=threshold)
        counts[i, :] = model_record.score()
        objective_values[i] = counts[i, 1] + counts[i, 2]
    best_i = numpy.argmin(objective_values)
    #print(f'{model_record.record_name} {thresholds[best_i]}')
    return thresholds[best_i], counts[best_i]


def main(argv=None):
    """Plot best thresholds on training data against various statistics

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    model_records = dict((name, utilities.ModelRecord(args.a_model_path, name))
                         for name in args.record_names)

    count_sum = 0
    for name, model_record in model_records.items():
        threshold, count = minimum_error(model_record)
        count_sum += count
    fraction = (count_sum[1] + count_sum[2]) / count_sum.sum()
    print(f'{count_sum} {count_sum[1]+count_sum[2]} {fraction}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
