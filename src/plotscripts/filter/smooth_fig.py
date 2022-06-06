""" smooth_fig.py <data> <plot_file>
"""
import sys
import argparse
import pickle

import numpy
import numpy.linalg

import plotscripts.utilities
import filter_fig


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

    # Read the data
    data = pickle.load(open(args.data, 'rb'))
    forward_means = data['forward_means'][:, 0]
    smooth_means = data['smooth_means'][:, 0]
    backward_informations = data['backward_informations']
    backward_means = data['backward_means'][:, 0]
    x_0 = data['x_coarse'][:, 0]
    t_ = numpy.array(range(len(x_0))) * data['dt_coarse']

    # Calculate backward means and covariances
    n_t = len(backward_informations)
    backward_covariances = numpy.zeros(backward_informations.shape)
    for t in range(n_t):
        # Use pseudo-inverse because near n_t backward_informations[t]
        # is singular
        backward_covariances[t] = numpy.linalg.pinv(backward_informations[t],
                                                    rcond=1e-8)

    # Set up axes
    fig, ((forward, forward_error), (backward, backward_error),
          (smooth, smooth_error)) = pyplot.subplots(nrows=3,
                                                    ncols=2,
                                                    figsize=(6, 10))
    all_axes = (forward, forward_error, backward, backward_error, smooth,
                smooth_error)
    for axis in all_axes:
        axis.set_ylim(-45, 45)
    forward.get_shared_x_axes().join(*all_axes)
    forward.get_shared_y_axes().join(*all_axes)

    # Plot forward filter results
    forward.plot(t_, forward_means, label='forwards')
    forward.plot(t_, x_0, label='$x_0$')
    filter_fig.plot_error(forward_error, t_, data['forward_covariances'],
                          x_0 - forward_means, 'forward error')

    # Plot backward filter results
    backward.plot(t_, backward_means, label='backwards')
    filter_fig.plot_error(backward_error, t_[:-1], backward_covariances[:-1],
                          (backward_means - x_0)[:-1], 'backward error')

    # Plot results of smoothing
    smooth.plot(t_, smooth_means, label='smooth')
    filter_fig.plot_error(smooth_error, t_, data['smooth_covariances'],
                          smooth_means - x_0, 'smooth error')

    # Add legends and remove some tick labels from the plots
    for axis in all_axes:
        axis.legend()

    for axis in (forward, forward_error, backward, backward_error):
        axis.set_xticklabels([])

    for axis in (forward_error, backward_error, smooth_error):
        axis.set_yticklabels([])

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
