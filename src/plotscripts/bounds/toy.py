"""plot.py options <data>.  Make figures that illustrate extended
Kalman filters

"""

import sys
import argparse
import pickle

import numpy
import scipy.linalg

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Figures to illustrate extended Kalman filter')
    parser.add_argument('--ToyTS1', type=str, help='path for result')
    parser.add_argument('--ToyStretch', type=str, help='path for result')
    parser.add_argument('--t_start',
                        type=int,
                        default=0,
                        help='start of interval to plot')
    parser.add_argument('--t_stop',
                        type=int,
                        default=120,
                        help='end of interval to plot')
    parser.add_argument('--t_view',
                        type=int,
                        default=50,
                        help='t for plot of forecast and update')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data', type=str, help='Path to data')
    return parser.parse_args(argv)


def time_series(args: argparse.Namespace, pyplot, data):
    """Make a stack of 3 plots: y[t], errors, log_probability

    Args:
        args:  Command line arguments
        pyplot:
        data: Dict with data to plot

    Return: figure
    """
    y = data['y'][:, 0]
    error = data['y_means'] - y
    y_deviations = numpy.sqrt(data['y_variances'])

    figure, (y_plot, error_plot, log_prob_plot) = pyplot.subplots(nrows=3,
                                                                  figsize=(6,
                                                                           5),
                                                                  sharex=True)
    # Drop tick labels on some shared axes.
    for axis in (y_plot, error_plot):
        axis.set_xticklabels([])

    y_plot.plot(y[args.t_start:args.t_stop], label=r'$y[t]$')

    error_plot.plot(y_deviations[args.t_start:args.t_stop],
                    label=r'$\sigma_y[t]$')
    error_plot.plot(error[args.t_start:args.t_stop], label=r'$y[t] - \mu_y[t]$')

    log_prob_plot.plot(data['log_probabilities'][args.t_start:args.t_stop],
                       label=r'$\log(\rm{Prob}(y[t]))$')
    log_prob_plot.set_xlabel('$t$')

    for axes in (y_plot, error_plot, log_prob_plot):
        axes.legend()

    return figure


def ellipse(mean, covariance, i_a=0, i_b=2):
    r""" Calculate points on x^T \Sigma^{-1} x = 1

    Args:
        mean: 3-vector
        covariance: 3x3 array
        i_a: index of first component
        i_b: index of second component
    """
    mean_2 = numpy.array([mean[i_a], mean[i_b]])
    covariance_2 = numpy.array([[covariance[i_a, i_a], covariance[i_a, i_b]],
                                [covariance[i_b, i_a], covariance[i_b, i_b]]])
    sqrt_cov_2 = scipy.linalg.sqrtm(covariance_2)
    n_points = 100
    theta = numpy.linspace(0, 2 * numpy.pi, n_points, endpoint=True)
    z = numpy.array([numpy.sin(theta), numpy.cos(theta)]).T
    result = numpy.dot(z, sqrt_cov_2) + mean_2
    return result


def find_ranges(forecast_a, forecast_b, update_a, update_b):
    """Find x_range and y_range for making two plots with the same
    scale in both x and y directions.  The goal is to illustrate
    stretching.

    Args:
        forecast_a: Points on ellipse
        forecast_b: Points on ellipse
        update_a: Points on ellipse
        update_b: Points on ellipse

    Returns: [range_a, range_b]

    The form of range in the return is:
    [ [x-Dx, x+Dx],
      [y-Dy, y+Dy]
    ]

    """

    def center_delta(forecast, update):
        """Find center and width/2 in both directions

        Args:
            forecast: Points on ellipse some time t
            update: Points on ellipse for the same t

        Returns: ([center_x, center_y], [delta_x, delta_y])
        """
        joined = numpy.concatenate((forecast, update))
        mins = joined.min(axis=0)
        maxs = joined.max(axis=0)
        centers = (mins + maxs) / 2
        deltas = (maxs - mins) / 2
        return centers, deltas

    center_a, delta_a = center_delta(forecast_a, update_a)
    center_b, delta_b = center_delta(forecast_b, update_b)
    delta_x = max(delta_a[0], delta_b[0]) * 1.1
    delta_y = max(delta_a[1], delta_b[1]) * 1.1
    delta = numpy.array([delta_x, delta_y])

    def new_range(center, delta):
        """
        Args:
            center: [x,y]
            delta: [Dx,Dy]

        Returns: [[x-Dx, y-Dy],[x+Dx, y+Dy]].T
        """
        return numpy.array([center - delta, center + delta]).T

    return [new_range(center_a, delta), new_range(center_b, delta)]


def stretch(args: argparse.Namespace, pyplot, data):
    """Make a figure to illustrate stretching of phase space

    """
    times = (args.t_view, args.t_view + 1)

    forecast_ellipses = [
        ellipse(data['forecast_means'][t], data['forecast_covariances'][t])
        for t in times
    ]
    update_ellipses = [
        ellipse(data['update_means'][t], data['update_covariances'][t])
        for t in times
    ]

    figure, both = pyplot.subplots(ncols=2, figsize=(12, 6))
    assert len(both) == 2
    # assert isinstance(both[0], matplotlib.axes._subplots.AxesSubplot)

    ranges = find_ranges(*forecast_ellipses, *update_ellipses)
    for n_axes, axes in enumerate(both):
        axes.plot(forecast_ellipses[n_axes][:, 0], forecast_ellipses[n_axes][:,
                                                                             1])
        axes.plot(update_ellipses[n_axes][:, 0], update_ellipses[n_axes][:, 1])
        axes.set_xlim(ranges[n_axes][0])
        axes.set_ylim(ranges[n_axes][1])

    for axes in both:
        axes.set_ylabel(r'$x_2$')
        axes.set_xlabel(r'$x_0$')

    return figure


def main(argv=None):
    """Make time series picture with fine, coarse, filtered data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    with open(args.data, 'rb') as file_:
        data = pickle.load(file_)

    if args.ToyTS1:
        figure = time_series(args, pyplot, data)
        if args.show:
            pyplot.show()
        figure.savefig(args.ToyTS1)

    if args.ToyStretch:
        figure = stretch(args, pyplot, data)
        if args.show:
            pyplot.show()
        figure.savefig(args.ToyStretch)

    return 0


if __name__ == "__main__":
    sys.exit(main())
