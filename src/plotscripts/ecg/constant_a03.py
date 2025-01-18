"""constant_a03.py A custom figure for ds23.pdf illustrating fixed duration of PQRST

constant_a03.py ecg_dir 

"""
import sys
import argparse
import pickle
import os

import numpy
import pint

import plotscripts.utilities

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Plot ECG and decoded states')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('ecg_file', type=str, help='Path to data')
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make time series picture with ecg and state data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    fig, ((aa,ab),(ba,bb)) = pyplot.subplots(nrows=2, ncols=2, figsize=(8, 5))
    # aa ab
    # ba bb
    with open(args.ecg_file, 'rb') as _file:
        _dict = pickle.load(_file)
        ecg = _dict['ecg']
        ecg_times = _dict['times'] * PINT('seconds')

    delta_a = 0.1  # Interval length in seconds for left column
    delta_b = 0.012  # Interval length in seconds for right column
    t_aa = 56.1  # Start time for upper left
    t_ba = 56.3  # Start time for lower left
    t_ab = 56.1083 # Start time for upper right
    t_bb = 56.346 # Start time for lower right
    def plot(axes, start, delta, linewidth=1, color='b'):
        n_start, n_stop = numpy.searchsorted(
            ecg_times.to('minutes').magnitude, (start, start+delta))
        times = 60 * (ecg_times[n_start:n_stop].to('minutes').magnitude - 56)
        axes.plot(times, ecg[n_start:n_stop], linewidth=linewidth, color=color)
    plot(aa, t_aa, delta_a)
    plot(ba, t_ba, delta_a)
    plot(ab, t_ab, delta_b, color='r')
    plot(bb, t_bb, delta_b, color='r')
    plot(aa, t_ab, delta_b, color='r')
    plot(ba, t_bb, delta_b, color='r')
        

    for axes in (ba, bb):
        axes.set_xlabel(r'($t-$00:56:00)/seconds')
    for axes in (aa, ba):
        axes.set_ylabel(r'$a03$ ecg/mV')
    bb.text(20.82, 0.215, 'P')
    bb.text(20.925, -0.9, 'Q')
    bb.text(20.96, 2.74, 'R')
    bb.text(21.01, -0.4, 'S')
    bb.text(21.29, 0.10, 'T')
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
