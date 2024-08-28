"""a01c02_states.py A custom figure for ds23.pdf with data from two records

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
    parser.add_argument('ecg_dir', type=str, help='Path to data')
    parser.add_argument('a01state_file', type=str, help='Path to data')
    parser.add_argument('c02state_file', type=str, help='Path to data')
    parser.add_argument('t_start', type=float, help="Time in minutes")
    parser.add_argument('t_stop', type=float, help="Time in minutes")
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make time series picture with ecg and state data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    fig, axeses_2x2 = pyplot.subplots(nrows=2, ncols=2, figsize=(6, 6))
    axeses = axeses_2x2.flatten()[[0, 2, 1, 3]]
    for i, name in enumerate('a01 c02'.split()):

        with open(os.path.join(args.ecg_dir, f'{name}'), 'rb') as _file:
            _dict = pickle.load(_file)
            ecg = _dict['ecg']
            ecg_times = _dict['times'] * PINT('seconds')

        t_start = args.t_start * PINT('minutes')
        t_stop = args.t_stop * PINT('minutes')
        n_start, n_stop = numpy.searchsorted(
            ecg_times.to('minutes').magnitude, (args.t_start, args.t_stop))
        times = ecg_times[n_start:n_stop].to('minutes').magnitude
        axeses[i * 2].plot(times, ecg[n_start:n_stop], label=name)
        axeses[i * 2].legend()

    for i, path in enumerate((args.a01state_file, args.c02state_file)):
        with open(path, "rb") as _file:
            states = pickle.load(_file)
            axeses[i * 2 + 1].plot(times, states[n_start:n_stop])

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
