"""plot.py: Illustrate performance of filter.
"""

import sys
import argparse
import pickle

import numpy
import scipy.stats

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Explore 1-d map')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data', type=str, help='path to data file')
    parser.add_argument('fig_path', type=str, help='path to figure')
    return parser.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    data = pickle.load(open(args.data, 'rb'))

    delta_ys = data['delta_ys'].reshape(-1)
    sorted = numpy.sort(delta_ys)
    n_data = len(sorted)
    y = numpy.arange(len(sorted)) / len(sorted)

    fig = pyplot.figure()
    ax = fig.add_subplot()
    a, b = scipy.stats.probplot(sorted, fit=True, plot=ax)
    mean = numpy.mean(sorted)
    dev = numpy.std(sorted)
    print(f'mean={mean}, standard_deviation={dev} {b}')
    #ax.plot(sorted, y)

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
