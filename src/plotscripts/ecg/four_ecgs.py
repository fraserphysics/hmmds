""" four_ecgs.py Plots of 4 ecgs

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
    parser.add_argument('--records',
                        type=str,
                        nargs=4,
                        default='a03 a10 b03 c02'.split())
    parser.add_argument('--time_interval',
                        type=float,
                        nargs=2,
                        default=[200.12, 200.18])
    parser.add_argument('--y_range', type=float, nargs=2, default=[-2.5, 3])
    parser.add_argument('ecg_dir', type=str, help='Path to data')
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make 4 time ecg series pictures

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    fig, axeses_2x2 = pyplot.subplots(nrows=2,
                                      ncols=2,
                                      sharex=True,
                                      sharey=True,
                                      figsize=(6, 6))
    pyplot.setp(
        axeses_2x2,
        xticks=args.time_interval,
        xticklabels=['200.12', '200.18'],
        yticks=args.y_range,
    )
    for axes in axeses_2x2[1, :]:
        axes.set_xlabel('time/minute')
    for axes in axeses_2x2[:, 0]:
        axes.set_ylabel('ECG')
    t_start, t_stop = args.time_interval
    y_min, y_max = args.y_range
    for axes, name in zip(axeses_2x2.flatten(), args.records):
        with open(os.path.join(args.ecg_dir, f'{name}'), 'rb') as _file:
            _dict = pickle.load(_file)
            ecg = _dict['ecg']
            ecg_times = _dict['times'] * PINT('seconds')
        n_start, n_stop = numpy.searchsorted(
            ecg_times.to('minutes').magnitude, (t_start, t_stop))
        times = ecg_times[n_start:n_stop].to('minutes').magnitude
        axes.plot(times, ecg[n_start:n_stop], label=name)
        axes.set_ylim(y_min, y_max)
        axes.legend()

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
