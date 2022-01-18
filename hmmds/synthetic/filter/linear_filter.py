""" plot_time_series.py <data> <plot_file>
"""
import sys
import argparse
import pickle

import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='plot_linear_simulation.pdf')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data', type=str, help='Path to data')
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make time series picture with fine, coarse, and quantized Lorenz
    data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    data = pickle.load(open(args.data, 'rb'))

    fig, ((axis_x_short, axis_y_long), (axis_x_01_short, axis_x0_long),
          (axis_y_short, axis_error)) = pyplot.subplots(nrows=3,
                                                        ncols=2,
                                                        figsize=(6, 10))

    axis_x_01_short.sharex(axis_y_short)
    axis_y_long.sharex(axis_error)
    axis_x0_long.sharex(axis_error)

    # Phase portrait
    axis_x_short.plot(data['x_fine'][:, 0], data['x_fine'][:, 1], label='$x$')
    # Both components vs time
    axis_x_01_short.plot(data['x_fine'][:, 0], label='$x_0$')
    axis_x_01_short.plot(data['x_fine'][:, 1], label='$x_1$')
    # Observations short-fine time
    axis_x_01_short.plot(list(range(len(data['x_fine'])))[::10],
                         data['x_fine'][::10, 0],
                         marker='.',
                         markersize=8,
                         linestyle='None')
    axis_y_short.plot(data['y_fine'])
    axis_y_short.plot(list(range(len(data['y_fine'])))[::10],
                      data['y_fine'][::10],
                      marker='.',
                      markersize=8,
                      linestyle='None',
                      label=r'$y$ short')
    # Observations long-coarse time
    axis_y_long.plot(data['y_coarse'], label='$y_\mathrm{long}$')
    # x_0 component and filtered estimate
    axis_x0_long.plot(data['x_coarse'][:, 0], label='$x_0$')
    axis_x0_long.plot(data['means'][:, 0], label='filtered')
    # Error of filter estimate and calculated variance of filter
    axis_error.plot(data['x_coarse'][:, 0] - data['means'][:, 0], label='error')
    sigma = numpy.sqrt(data['covariances'][:, 0, 0])
    axis_error.plot(sigma, color='red', label='$\pm\sigma$')
    axis_error.plot(-sigma, color='red')

    for axis in (axis_x_short, axis_x_01_short, axis_y_short, axis_y_long,
                 axis_x0_long, axis_error):
        axis.legend()

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
