""" state_sequence.py <data> <plot_file>
"""
import sys
import argparse

import numpy

import plotscripts.utilities

def parse_args(argv):
    """ Convert command line arguments into a namespace
    """
    parser = argparse.ArgumentParser(description='Plot sequence of integer states')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--n_samples', type=int, default=100, help="length of y axis")
    parser.add_argument('data', type=str, help="path to state sequence")
    parser.add_argument('fig_path', type=str, help="path to result")
    return parser.parse_args(argv)

def main(argv=None):
    """ plot a sequence of states
    """
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    data = numpy.empty((args.n_samples, 2), dtype=int)
    with open(args.data, 'r', encoding='utf-8') as _file:
        for data_i in data:
            data_i[:] = [int(x) for x in _file.readline().split()]

    fig, axes = pyplot.subplots(1,1,figsize=(6,2))
    axes.plot(data[:,0], data[:,1],'kd')
    axes.set_xlabel(r'$t$')
    axes.set_ylabel(r'$s(t)$')
    axes.set_ylim(-.5, 12)
    axes.set_yticks(numpy.arange(0,12.1,2))
    axes.set_xticks(numpy.arange(0,100.1,20))

    fig.tight_layout()
    fig.savefig(args.fig_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
