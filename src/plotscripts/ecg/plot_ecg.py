"""plot_ecg.py Plot segments of ecg file on screen

plot_ecg.py segments of ecg_file t_window t_start_0 t_start_1 ...

ecg_file   Rtimes/a01.ecg
t_window   0.03            In minutes
t_start    70.81 71.11

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

    parser = argparse.ArgumentParser(description='Plot segments of ECG file')
    parser.add_argument('--show',
                        action='store_false',
                        help="display figure using Qt5")
    parser.add_argument('ecg_file', type=str, help='Path to data')
    parser.add_argument('t_window', type=float, help="Time in minutes")
    parser.add_argument('t_starts',
                        nargs="+",
                        type=float,
                        help="Time in minutes")
    return parser.parse_args(argv)


def main(argv=None):
    """Make time series picture with ecg and state data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    print(f"{type(args.t_starts)}")
    with open(args.ecg_file, 'rb') as _file:
        _dict = pickle.load(_file)
        ecg = _dict['raw']
        ecg_times = _dict['times'] * PINT('seconds')

    n_segments = len(args.t_starts)

    fig, axeses = pyplot.subplots(nrows=n_segments, figsize=(6, 8), sharex=True)
    if n_segments == 1:
        axeses = [axeses]

    for n, axes in enumerate(axeses):
        n_start, n_stop = numpy.searchsorted(
            ecg_times.to('minutes').magnitude,
            (args.t_starts[n], args.t_starts[n] + args.t_window))
        #times = ecg_times[n_start:n_stop].to('minutes').magnitude
        times = numpy.arange(n_start, n_stop)
        axes.plot(times, ecg[n_start:n_stop])

    pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
