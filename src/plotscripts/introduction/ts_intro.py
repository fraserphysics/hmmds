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

    fig = pyplot.figure(figsize=(6, 4))

    x_fine = plotscripts.utilities.Axis(data=fine[0],
                                        magnitude=False,
                                        label=r'$\tau$')
    y_fine = plotscripts.utilities.Axis(data=fine[1],
                                        magnitude=False,
                                        ticks=numpy.arange(-10, 10.1, 10),
                                        label=r'$x_1(\tau)$')

    # Initialize the subplots.
    upper = plotscripts.utilities.sub_plot(fig, (2, 1, 1),
                                           x_fine,
                                           y_fine,
                                           color='b')
    lower = fig.add_subplot(2, 1, 2)

    upper.plot(coarse[0], coarse[1], 'ro')
    upper.set_ylim(-17, 17)
    upper.set_xlim(0, 6)

    lower.plot(quantized[0], quantized[1], 'kd')
    lower.set_xlabel(r'$t$')
    lower.set_ylabel(r'$y(t)$')
    lower.set_ylim(0.5, 4.5)
    lower.set_yticks(numpy.arange(1, 4.1, 1))
    lower.set_xticks(numpy.arange(0, 40.1, 10))

    fig.subplots_adjust(hspace=0.3)

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
