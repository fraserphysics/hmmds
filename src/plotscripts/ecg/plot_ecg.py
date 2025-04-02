"""plot_ecg.py Plot segments of ecg file on screen

plot_ecg.py segments of ecg_file t_window t_start_0 t_start_1 ...

ecg_file   build/derived_data/apnea/ecgs/a01
t_window   0.03            In minutes
t_start    70.81 71.11

"""
import sys
import argparse
import pickle

import numpy
import pint

import plotscripts.utilities
import ecg_grid

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Plot segments of ECG file')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('ecg_file', type=str, help='Path to data')
    parser.add_argument('t_window', type=float, help="Time in minutes")
    parser.add_argument('t_starts',
                        nargs="+",
                        type=float,
                        help="Time in minutes")
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make time series picture with ecg and state data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    with open(args.ecg_file, 'rb') as _file:
        _dict = pickle.load(_file)
        ecg = _dict['ecg']
        ecg_times = _dict['times'] * PINT('seconds')

    n_segments = len(args.t_starts)

    y_min = -3
    y_max = 4
    fig, axeses = pyplot.subplots(nrows=n_segments,
                                  figsize=(6, 3.1 * n_segments))
    if n_segments == 1:
        axeses = [axeses]

    for n, axes in enumerate(axeses):
        axes.set_xlabel('time H:M:S')
        axes.set_ylabel('ECG/mV')
        n_start, n_stop = numpy.searchsorted(
            ecg_times.to('minutes').magnitude,
            (args.t_starts[n], args.t_starts[n] + args.t_window))
        times = ecg_times[n_start:n_stop].to('seconds').magnitude
        axes.plot(times, ecg[n_start:n_stop])
        ecg_grid.decorate(axes, times[0], times[-1], y_min, y_max)

    fig.tight_layout()
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
