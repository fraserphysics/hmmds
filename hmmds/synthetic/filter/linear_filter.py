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
    for name in 'x_fine y_fine x_coarse y_coarse filtered_coarse'.split():
        parser.add_argument(name, type=str, help='Path to data')
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def read_filtered(name):
    """Read a file with means and covariances and return arrays of floats.
    """
    mean_list = []
    covariance_list = []
    with open(name, mode='r', encoding='utf-8') as file:
        for line in file.readlines():
            parts = line.split()
            if len(parts) == 0:
                continue
            if len(parts) == 3:
                mean_list.append([float(parts[1]), float(parts[2])])
            if len(parts) == 2:
                covariance_list.append([float(parts[0]), float(parts[1])])
    n_samples = len(mean_list)
    assert n_samples*2 == len(covariance_list)
    means = numpy.array(mean_list)
    covariances = numpy.array(covariance_list).reshape((n_samples, 2, 2))
    return means, covariances

def read_array(name):
    """Read a simple text file and return an array of floats.
    """
    with open(name, mode='r', encoding='utf-8') as file:
        return numpy.array([
            [float(x) for x in line.split()] for line in file.readlines()
        ])

def main(argv=None):
    """Make time series picture with fine, coarse, and quantized Lorenz
    data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    x_fine = read_array(args.x_fine)
    y_fine = read_array(args.y_fine)
    x_coarse = read_array(args.x_coarse)
    y_coarse = read_array(args.y_coarse)
    filtered_mean, filtered_covariance = read_filtered(args.filtered_coarse)

    fig, ((axis_x_short, axis_y_long), (axis_x_01_short, axis_x0_long), (axis_y_short, axis_error)) = pyplot.subplots(nrows=3, ncols=2, figsize=(6, 10))

    axis_x_01_short.sharex(axis_y_short)
    axis_y_long.sharex(axis_error)
    axis_x0_long.sharex(axis_error)

    # Phase portrait
    axis_x_short.plot(x_fine[:,0], x_fine[:,1], label='$x$')
    # Both components vs time
    axis_x_01_short.plot(x_fine[:,0], label='$x_0$')
    axis_x_01_short.plot(x_fine[:,1], label='$x_1$')
    # Observations short-fine time
    axis_x_01_short.plot(list(range(len(x_fine)))[::10], x_fine[::10,0], marker='.', markersize=8, linestyle='None')
    axis_y_short.plot(y_fine)
    axis_y_short.plot(list(range(len(y_fine)))[::10], y_fine[::10], marker='.', markersize=8, linestyle='None', label=r'$y$ short')
    # Observations long-coarse time
    axis_y_long.plot(y_coarse, label='$y_\mathrm{long}$')
    # x_0 component and filtered estimate
    axis_x0_long.plot(x_coarse[:,0], label='$x_0$')
    axis_x0_long.plot(filtered_mean[:,0], label='filtered')
    # Error of filter estimate and calculated variance of filter
    axis_error.plot(x_coarse[:,0] - filtered_mean[:,0], label='error')
    sigma = numpy.sqrt(filtered_covariance[:,0,0])
    axis_error.plot(sigma, color='red', label='$\pm\sigma$')
    axis_error.plot(-sigma, color='red')

    for axis in (axis_x_short, axis_x_01_short, axis_y_short, axis_y_long, axis_x0_long, axis_error):
        axis.legend()

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
