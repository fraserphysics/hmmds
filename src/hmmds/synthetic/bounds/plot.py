"""plot.py options <data>.  Make figures that illustrate extended
Kalman filters

"""
# ToyTS1.pdf Figure 5.1 Plots: (1) y[t]; (2) sigma[t], error[t]; (3)
# log(P(y[t])).  Data in data_h_view

# ToyStretch.pdf Figure 5.2 Update and forecast distributions for
# t=105 & 106.  Data in data_h_view

# ToyH.pdf Figure 5.3 Two plots: (1) -^h vs measurement noise and
# sample time; (2) -^h vs sample time

# benettin.pdf Figure 5.5 Laypunov exponent calculations

# LikeLor.pdf Figure 5.6 Cross entropy vs number of states

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
                        default=105,
                        help='t for plot of forecast and update')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data', type=str, help='Path to data')
    return parser.parse_args(argv)


def plot_error(axis, sample_times, covariance, difference, label):
    axis.plot(sample_times, difference, label=label)
    sigma = numpy.sqrt(covariance[:, 0, 0])
    axis.plot(sample_times, 2 * sigma, color='red', label='$\pm2\sigma$')
    axis.plot(sample_times, -2 * sigma, color='red')


def time_series(args, pyplot, data):
    y = data['y'][:, 0]
    error = data['y_means'] - y
    y_deviations = numpy.sqrt(data['y_variances'])

    fig, (y_plot, error_plot, log_prob_plot) = pyplot.subplots(nrows=3,
                                                               figsize=(6, 5))
    # Force matching ticks
    log_prob_plot.get_shared_x_axes().join(y_plot, error_plot, log_prob_plot)
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

    return fig


def ellipse(mean, covariance, i_a=0, i_b=2):
    """ Calculate points on x^T \Sigma^{-1} x = 1

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
    """find x_range and y_range for plotting
    """

    def center_delta(forecast, update):
        """Find center and width/2 in both directions
        """
        joined = numpy.concatenate((forecast, update))
        mins = joined.min(axis=0)
        maxs = joined.max(axis=0)
        centers = (mins + maxs) / 2
        deltas = (maxs - mins) / 2
        return centers, deltas

    center_a, delta_a = center_delta(forecast_a, update_a)
    center_b, delta_b = center_delta(forecast_b, update_b)
    delta_x = max(delta_a[0], delta_b[0])*1.1
    delta_y = max(delta_a[1], delta_b[1])*1.1
    return [
        [
            [
                center_a[0] - delta_x,
                center_a[0] + delta_x,
            ],  # x_range_a
            [center_a[1] - delta_y, center_a[1] + delta_y]
        ],  # y_range_a
        [
            [center_b[0] - delta_x, center_b[0] + delta_x],  # x_range_b
            [center_b[1] - delta_y, center_b[1] + delta_y]
        ]  # y_range_b
    ]


def stretch(args, pyplot, data):
    times = (args.t_view, args.t_view + 1)

    forecast_ellipses = [
        ellipse(data['forecast_means'][t], data['forecast_covariances'][t])
        for t in times
    ]
    update_ellipses = [
        ellipse(data['update_means'][t], data['update_covariances'][t])
        for t in times
    ]

    fig, both = pyplot.subplots(ncols=2, figsize=(12, 6))
    assert len(both) == 2
    # assert isinstance(both[0], matplotlib.axes._subplots.AxesSubplot)

    ranges = find_ranges(*forecast_ellipses, *update_ellipses)
    for n, axes in enumerate(both):
        axes.plot(forecast_ellipses[n][:, 0], forecast_ellipses[n][:, 1])
        axes.plot(update_ellipses[n][:, 0], update_ellipses[n][:, 1])
        axes.set_xlim(ranges[n][0])
        axes.set_ylim(ranges[n][1])

    for axes in both:
        axes.set_ylabel(r'$x_2$')
        axes.set_xlabel(r'$x_0$')

    return fig


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
