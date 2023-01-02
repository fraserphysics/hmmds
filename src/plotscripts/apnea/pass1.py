""" pass1.py: Creates Fig. 6.8 of the book

python pass1.py pass1_report pass1.pdf
"""

import sys
import argparse
import pickle
import os

import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Plot classification of records in first pass.')
    parser.add_argument('--show',
                        action='store_true',
                        help='display figure in pop-up window')
    parser.add_argument('report', type=str, help='Path to input data')
    parser.add_argument('output', type=str, help='Path to result')
    args = parser.parse_args(argv)
    return args


def lines(ax):
    """Make arrays for lines
    """
    x = numpy.array([-2.0, 3.0])
    low_line = 1.82
    high_line = 2.6
    ax.plot(x,low_line - .5 * x, 'm-', label=r'$R+\frac{llr}{2}=%4.2f$' % low_line)
    ax.plot(x, high_line - .5 * x, 'k-', label=r'$R+\frac{llr}{2}=%4.2f$' % high_line)
def main(argv=None):
    '''Call with arguments: report, fig_file

    fig_file is a path where this script writes the result.

    '''
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, matplotlib, pyplot = plotscripts.utilities.import_and_parse(
        parse_args, argv)
    fig, ax = pyplot.subplots(figsize=(8, 4))

    with open(args.report, 'rb') as _file:
        reports = pickle.load(_file)
    classes = {key: [] for key in 'a b c x'.split()}
    for name, report in reports.items():
        a_per = report['a_ll']/report['n_times']
        n_per = report['n_ll']/report['n_times']
        classes[name[0]].append((a_per, n_per))

    ax.set_xlabel('$llr_a$')
    ax.set_ylabel('$llr_n$')
    sym = {'a': 'rs', 'b': 'go', 'c': 'bD', 'x': 'mx'}
    for key, value in classes.items():
        pair = value[0]
        ax.plot(pair[0], pair[1], sym[key], label=key)
    #lines(ax)
    #ax.legend(loc='upper right')
    ax.legend()
    for key, value in classes.items():
        for pair in value[1:]:
            ax.plot(pair[0], pair[1], sym[key], label=key)
    # ax.legend()
    fig.savefig(args.output)
    if args.show:
        pyplot.show()
    return 0


if __name__ == "__main__":
    args = ['pass1_report', 'pass1.pdf']
    sys.exit(main())
# Local Variables:
# mode: python
# End:
