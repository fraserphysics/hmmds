"""plot_simulation.py Plot result of HMM.simulate(n_simulate)

plot_simulation.py hmm_file n_simulate target.pdf

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

    parser = argparse.ArgumentParser(description='Simulate and plot ECG signal')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('hmm', type=str, help='Path to data')
    parser.add_argument('n_simulate',
                        type=int,
                        help='Number of samples to simulate')
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Simulate and plot ECG

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    with open(args.hmm, 'rb') as _file:
        _, _hmm = pickle.load(_file)

    states, ecg = _hmm.simulate(args.n_simulate)
    times = numpy.arange(args.n_simulate) / 100

    fig, axes = pyplot.subplots(nrows=2, sharex='all', figsize=(5, 4))
    axes[0].plot(times, ecg)
    axes[0].set_ylabel('ecg')
    axes[1].plot(times, states)
    axes[1].set_ylabel('state')
    axes[1].set_xlabel('seconds')

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
