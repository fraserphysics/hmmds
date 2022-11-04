""" benettin_plot.py Make figure to illustrate calculating Lyapunov exponents.
"""

import sys
import argparse
import pickle

import numpy
import scipy.linalg

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Figure to illustrate calculating Lyapunov exponents')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data', type=str, help='Path to input data')
    parser.add_argument('figure', type=str, help='Path to result')
    return parser.parse_args(argv)


def main(argv=None):
    """Make time series picture with fine, coarse, filtered data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    figure, (top_axes, bottom_axes) = pyplot.subplots(nrows=2, figsize=(6, 6))

    with open(args.data, 'rb') as file_:
        data = pickle.load(file_)
    cumsum = numpy.cumsum(data, axis=1) / 0.15
    n_runs, n_times, three = data.shape
    assert three == 3
    times = numpy.arange(1.0, n_times + .5, 1.0)

    for n_run in range(n_runs):
        top_axes.plot(times, cumsum[n_run, :, 0] / times)

    if args.show:
        pyplot.show()
    figure.savefig(args.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
