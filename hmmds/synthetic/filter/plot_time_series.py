""" plot_time_series.py <data> <plot_file>
"""
import sys
import argparse

import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='plot_linear_simulation.pdf')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('x_path', type=str, help="path to data")
    parser.add_argument('y_path', type=str, help="path to data")
    parser.add_argument('filtered', type=str, help="path to data")
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
        with open(name, mode='r', encoding='utf-8') as file:
            return numpy.array(
                [[float(x) for x in line.split()] for line in file.readlines()])

    x_data = read_data(args.x_path)
    y_data = read_data(args.y_path)
    filtered_data = read_data(args.filtered)
    fig, (axis_x, axis_y, axis_filtered_0,
          axis_filtered_1) = pyplot.subplots(nrows=4, ncols=1, figsize=(6, 15))

    axis_filtered_0.sharex(axis_filtered_1)

    axis_x.plot(x_data[:, 0], x_data[:, 1], label='x')
    axis_y.plot(y_data, label='y')
    axis_filtered_0.plot(x_data[:, 0], label='$x_0$')
    axis_filtered_0.plot(filtered_data[:, 0], label='filtered')
    axis_filtered_1.plot(x_data[:, 1], label='$x_1$')
    axis_filtered_1.plot(filtered_data[:, 1], label='filtered')

    for axis in (axis_x, axis_y, axis_filtered_0, axis_filtered_1):
        axis.legend()

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
