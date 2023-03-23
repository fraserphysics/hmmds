"""ecg_states_fig.py Plot ecg and states from Viterbi decoding.

ecg_states_fig.py ecg_file state_file  t_start t_stop result

ecg_file   Rtimes/a01.ecg
state_file ECG/AR1k20/states_a01
t_start    70.1
t_stop     70.2
result     foo.pdf
"""
import sys
import argparse
import pickle

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
    parser.add_argument('state_file', type=str, help='Path to data')
    parser.add_argument('t_start', type=float, help="Time in minutes")
    parser.add_argument('t_stop', type=float, help="Time in minutes")
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make time series picture with ecg and state data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    with open(args.ecg_file, 'rb') as _file:
        _dict = pickle.load(_file)
        ecg = _dict['raw']
        ecg_times = _dict['times'] * PINT('seconds')

    with open(args.state_file, "rb") as _file:
        states = pickle.load(_file)

    t_start = args.t_start * PINT('minutes')
    t_stop = args.t_stop * PINT('minutes')

    n_start, n_stop = numpy.searchsorted(
        ecg_times.to('minutes').magnitude, (args.t_start, args.t_stop))

    fig, (ecg_axes, states_axes) = pyplot.subplots(nrows=2, figsize=(6, 8))

    times = ecg_times[n_start:n_stop].to('minutes').magnitude
    ecg_axes.plot(times, ecg[n_start:n_stop])
    states_axes.plot(times, states[n_start:n_stop])

    # Legends for all axes
    for axis in (ecg_axes, states_axes):
        axis.legend()

    # Force matching ticks
    ecg_axes.get_shared_x_axes().join(ecg_axes, states_axes)

    # Drop tick labels
    ecg_axes.set_xticklabels([])

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
