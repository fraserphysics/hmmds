"""comparision_plot.py Plot error rate for list of models

python comparison_plot.py input.pkl result.pdf --show

"""
import sys
import argparse
import typing
import pickle
import os.path

import numpy
import scipy.optimize

import plotscripts.utilities


def parse_args(argv):
    """ Parse command line arguments
    """

    parser = argparse.ArgumentParser("Plot error rate for list of models")
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('input',
                        type=argparse.FileType('rb'),
                        help='Calculated error rates')
    parser.add_argument('figure_path',
                        nargs='?',
                        type=str,
                        default='compare_models.pdf',
                        help='Write result to this path')
    args = parser.parse_args(argv)
    return args


def plot(axes, error_counts, xlabel=None):
    x = []
    error_rate = []
    false_alarm = []
    missed_detection = []
    for x_, counts in error_counts.items():
        x.append(float(x_))
        false_alarm.append(counts[1])
        missed_detection.append(counts[2])
        error_rate.append(counts[1] + counts[2])
    axes.plot(x, false_alarm, label="false alarm")
    axes.plot(x, missed_detection, label="missed detection")
    axes.plot(x, error_rate, label="all errors")
    axes.legend()
    if xlabel:
        axes.set_xlabel(xlabel)
    axes.set_ylabel('Number of errors')


def main(argv=None):
    """Plot pass2 classification performance against a parameter

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    _dict = pickle.load(args.input)

    fig, axes = pyplot.subplots(nrows=1, figsize=(6, 4))
    plot(axes, _dict['error_counts'], _dict['x_label'])

    fig.tight_layout()
    fig.savefig(args.figure_path)

    if args.show:
        pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
