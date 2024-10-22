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

    fig, axeses_2x2 = pyplot.subplots(nrows=2, ncols=2, figsize=(8, 5))
    axeses = axeses_2x2.flatten()[[0, 2, 1, 3]]
    with open(args.ecg_file, 'rb') as _file:
        _dict = pickle.load(_file)
        ecg = _dict['ecg']
        ecg_times = _dict['times'] * PINT('seconds')

    delta = 0.012
    for minute_start, minute_stop, axes in ((56.1, 56.2, axeses[0]),
                                            (56.3, 56.4, axeses[1]),
                                            (56.1083, 56.1083 + delta,
                                             axeses[2]), (56.346,
                                                          56.346 + delta,
                                                          axeses[3])):
        n_start, n_stop = numpy.searchsorted(
            ecg_times.to('minutes').magnitude, (minute_start, minute_stop))
        times = 60 * (ecg_times[n_start:n_stop].to('minutes').magnitude - 56)
        axes.plot(times, ecg[n_start:n_stop])

    for axes in (axeses[1], axeses[3]):
        axes.set_xlabel(r'($t-$00:56:00)/seconds')
    for axes in (axeses[0], axeses[1]):
        axes.set_ylabel(r'$a03$ ecg/mV')
    axeses[3].text(20.82, 0.215, 'P')
    axeses[3].text(20.925, -0.9, 'Q')
    axeses[3].text(20.96, 2.74, 'R')
    axeses[3].text(21.01, -0.4, 'S')
    axeses[3].text(21.29, 0.10, 'T')
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
