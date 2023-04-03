"""train_characteristic.py: Plot aposterior probability vs training iteration

plot_ecg.py segments of ecg_file t_window t_start_0 t_start_1 ...


"""
import sys
import argparse
import pickle

import numpy
import pint

import plotscripts.utilities
import utilities

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Plot progress of training')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('log_file', type=str, help='Path to data')
    parser.add_argument('fig_path', type=str, help="path to figure")

    return parser.parse_args(argv)


def main(argv=None):
    """Plot progress of training.

    """
    ecg_length = 2957000
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    column_dict = utilities.read_train_log(args.log_file)
    # keys are L0 -- L32, prior, U/n
    n_steps = len(column_dict['L0'])
    likelihood = numpy.zeros(n_steps)
    for key, value in column_dict.items():
        if key[0] == 'L':
            likelihood += value

    fig, axeses = pyplot.subplots(nrows=3, sharex='all', figsize=(6, 8))
    for i, (y,label) in enumerate((
            (likelihood/ecg_length, 'likelihood'),
            (column_dict['prior']/ecg_length, 'prior'),
            (column_dict['U/n'], 'MAP'))):
        axeses[i].plot(y, label=label)
        axeses[i].legend()
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    
    return 0


if __name__ == "__main__":
    sys.exit(main())
