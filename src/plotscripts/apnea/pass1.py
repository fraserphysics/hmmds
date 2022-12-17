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
    parser.add_argument('report',
                        type=str,
                        help='Path to input data')
    parser.add_argument('output', type=str, help='Path to result')
    args = parser.parse_args(argv)
    return args

def main(argv=None):
    '''Call with arguments: report, fig_file

    fig_file is a path where this script writes the result.

    '''
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    args, matplotlib, pyplot = plotscripts.utilities.import_and_parse(
        parse_args, argv)
    fig, ax = pyplot.subplots(figsize=(8,4))

    with open(args.report, 'rb') as _file:
        reports = pickle.load(_file)
    classes = {key:[] for key in 'a b c x'.split()}
    for report in reports:
        classes[report.name[0]].append((report.llr, report.r))

    ax.set_xlabel('$llr$')
    #ax.set_xlim(-2, 9)
    ax.set_ylabel('$R$')
    #ax.set_ylim(1.4, 3.2)
    sym = {'a':'rs', 'b':'go', 'c':'bD', 'x':'mx'}
    for key, value in classes.items():
        ax.plot(value[0], value[1], sym[key], label=key)
    x = numpy.array([-2.0, 3.0])
    low_line = 1.82
    high_line = 2.6
    y = low_line-.5*x
    ax.plot(x, y, 'm-', label=r'$R+\frac{llr}{2}=%4.2f$'%low_line)
    y = high_line-.5*x
    ax.plot(x, y, 'k-', label=r'$R+\frac{llr}{2}=%4.2f$'%high_line)
    #ax.legend(loc='upper right')
    ax.legend()
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
