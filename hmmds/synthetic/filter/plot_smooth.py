""" plot_time_series.py <data> <plot_file>
"""
import sys
import argparse
import pickle

import numpy

import plotscripts.utilities
import linear_filter


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
    """Make a figure illustrating backwards filtering and smoothing.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    data = pickle.load(open(args.data, 'rb'))

    fig, ((axis_x, axis_forward_error), (axis_backward, axis_backward_error),
          (axis_smooth, axis_smooth_error)) = pyplot.subplots(nrows=3,
                                                              ncols=2,
                                                              figsize=(6, 10))
    all_axes = (axis_x, axis_forward_error, axis_backward, axis_backward_error,
                axis_smooth, axis_smooth_error)
    for axis in all_axes:
        axis.set_ylim(-0.9, 0.9)
    axis_x.get_shared_x_axes().join(*all_axes)

    x_0 = data['x_coarse'][:, 0]
    forward = data['forward_means'][:, 0]
    backward = data['back_means'][:, 0]
    smooth = data['smooth_means'][:, 0]
    t_ = numpy.array(range(len(x_0))) * data['dt_coarse']

    axis_x.plot(t_, forward, label='forwards')
    axis_x.plot(t_, x_0, label='$x_0$')
    linear_filter.plot_error(axis_forward_error, t_,
                             data['forward_covariances'], x_0 - forward,
                             'forward error')

    axis_backward.plot(t_, backward, label='backwards')
    linear_filter.plot_error(axis_backward_error, t_[:-1],
                             data['back_covariances'][:-1],
                             backward[1:] - x_0[:-1], 'backward error')

    axis_smooth.plot(t_, smooth, label='smooth')
    linear_filter.plot_error(axis_smooth_error, t_, data['smooth_covariances'],
                             smooth - x_0, 'smooth error')

    for axis in all_axes:
        axis.legend()

    for axis in (axis_x, axis_forward_error, axis_backward,
                 axis_backward_error):
        axis.set_xticklabels([])

    for axis in (axis_forward_error, axis_backward_error, axis_smooth_error):
        axis.set_yticklabels([])

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
