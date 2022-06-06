""" pass1.py: Creates Fig. 6.8 of the book

Call with arguments pass1_report pass1.pdf

pass1_report: Path to a text file

pass1_pdf: Path to write result

"""

import sys
import argparse
import pickle

import numpy
import matplotlib

import hmmds.applications.apnea.utilities  # For pickle


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    if not argv:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='Make a plot of first pass classification')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('pass1_report', type=str, default="path to report")
    parser.add_argument('pass1_pdf', type=str, default="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Figure of first pass classifier.
    """

    if not argv:
        argv = sys.argv[1:]

    args = parse_args(argv)

    if args.show:
        matplotlib.use('Qt5Agg')
    else:
        matplotlib.use('PDF')  # Permits absence of enviroment variable DISPLAY
    # Must be after matplotlib.use
    import matplotlib.pyplot  #  pylint: disable=import-outside-toplevel, redefined-outer-name

    with open(args.pass1_report, 'rb') as _file:
        data = pickle.load(_file)
    params = {
        'axes.labelsize': 12,
        'text.fontsize': 10,
        #'legend.fontsize': 10,
        'text.usetex': True,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    }
    matplotlib.rcParams.update(params)

    fig = matplotlib.pyplot.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('$llr$')
    #ax.set_xlim(-2, 9)
    ax.set_ylabel('$R$')
    ax.set_ylim(1.4, 3.2)
    # ToDo: Use letters with colors?  Also want better legend
    sym = {'a': 'rs', 'b': 'go', 'c': 'bD', 'x': 'mx'}
    sym = {'Low': 'go', 'Medium': 'yx', 'High': 'rs'}
    for record in data:
        key = record.name[0]
        key = record.level
        ax.plot(record.llr, record.r, sym[key])
    x = numpy.array([-2.0, 3.0])
    low_line = 1.82
    high_line = 2.6
    y = low_line - .5 * x
    ax.plot(x, y, 'm-', label=r'$R+\frac{llr}{2}=%4.2f$' % low_line)
    y = high_line - .5 * x
    ax.plot(x, y, 'k-', label=r'$R+\frac{llr}{2}=%4.2f$' % high_line)
    ax.legend(loc='lower right')
    if args.show:
        matplotlib.pyplot.show()
    fig.savefig(args.pass1_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
# Local Variables:
# mode: python
# End:
