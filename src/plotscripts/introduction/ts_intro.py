""" ts_intro.py <fine> <coarse> <quantized> <plot_file>
"""
import sys
import argparse

import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Make GaussMix.pdf')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('fine_path',
                        type=str,
                        help="path to finely sampled data")
    parser.add_argument('coarse_path',
                        type=str,
                        help="path to coarsely sampled data")
    parser.add_argument('quantized_path',
                        type=str,
                        help="path to quantized data")
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make time series picture with fine, coarse, and quantized Lorenz
    data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    def read_data(name):
        """Read a text file and return an array of floats.
        """
        with open(name, 'r') as file:
            return numpy.array([
                [float(x) for x in line.split()] for line in file.readlines()
            ]).T

    fine = read_data(args.fine_path)
    coarse = read_data(args.coarse_path)
    quantized = read_data(args.quantized_path)

    fig, (upper, lower) = pyplot.subplots(nrows=2, figsize=(6, 4))

    upper.plot(fine[0], fine[1], color='b')
    upper.plot(coarse[0, :40], coarse[1, :40], 'ro')
    upper.set_xlabel(r'$\tau$')
    upper.set_ylabel(r'$x_0(\tau)$')
    upper.set_ylim(-17, 17)
    upper.set_xlim(-.15, 6.15)

    lower.plot(quantized[0, :40], quantized[1, :40], 'kd')
    lower.set_xlabel(r'$t$')
    lower.set_ylabel(r'$y(t)$')
    lower.set_ylim(-0.5, 3.5)
    lower.set_xlim(-1, 41)

    fig.tight_layout()
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
