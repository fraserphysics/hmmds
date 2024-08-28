"""plot_like.py Compare plots of ecg and log likelihood for two records

plot_like.py ecg_dir likelihood_dir t_start t_stop target.pdf

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
    parser.add_argument('like_dir', type=str, help='Path to data')
    parser.add_argument('t_interval',
                        type=float,
                        nargs=2,
                        help="Time in minutes")
    parser.add_argument('record_names', type=str, nargs=2, help="eg, a14 x07")
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Compare plots of ecg and log likelihood for two records

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    fig, axeses = pyplot.subplots(nrows=2, ncols=2, figsize=(6, 6))
    for i, name in enumerate(args.record_names):
        with open(os.path.join(args.ecg_dir, f'{name}'), 'rb') as _file:
            _dict = pickle.load(_file)
            ecg = _dict['ecg']
            ecg_times = _dict['times'] * PINT('seconds')
        with open(os.path.join(args.like_dir, name), 'rb') as _file:
            likelihood = pickle.load(_file)

        t_start = args.t_interval[0]
        t_stop = args.t_interval[1]
        n_start, n_stop = numpy.searchsorted(
            ecg_times.to('minutes').magnitude, (t_start, t_stop))
        times = ecg_times[n_start:n_stop].to('minutes').magnitude
        axeses[0, i].plot(times, ecg[n_start:n_stop], label=f'ecg_{name}')
        axeses[1, i].plot(times,
                          numpy.log(likelihood[n_start:n_stop]),
                          label=f'log prob_{name}')

    # Legends for all axes
    for axis in axeses.flatten():
        axis.legend()

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
