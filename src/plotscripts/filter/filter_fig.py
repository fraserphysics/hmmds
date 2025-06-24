"""filter_fig.py Make a 6 plot figure illustrating forward filtering.

python filter_fig.py data fig_path
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
    parser.add_argument('--sample_ratio',
                        type=int,
                        default=10,
                        help='ratio of fine to coarse')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data', type=str, help='Path to data')
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def plot_error(axis, sample_times, covariance, difference, label):
    axis.plot(sample_times, difference, label=label)
    sigma = numpy.sqrt(covariance[:, 0, 0])
    axis.plot(sample_times, 2 * sigma, color='red', label='$\pm2\sigma$')
    axis.plot(sample_times, -2 * sigma, color='red')


def main(argv=None):
    """Make time series picture with fine, coarse, filtered data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    data = pickle.load(open(args.data, 'rb'))
    t_fine = numpy.array(range(len(data['y_fine']))) * data['dt_fine']
    t_coarse = numpy.array(range(len(data['y_coarse']))) * data['dt_coarse']

    fig, ((phase_portrait, observations), (states, filtered),
          (sampled_observations, errors)) = pyplot.subplots(nrows=3,
                                                            ncols=2,
                                                            figsize=(6, 10))

    phase_portrait.plot(data['x_fine'][:, 0],
                        data['x_fine'][:, -1],
                        label='$x$')
    # All components vs time
    for i, x_i in enumerate(data['x_fine'].T):
        states.plot(t_fine, x_i, label=f'$x_{i}$')
    # Plot states vs time.  Short time finely sampled
    states.plot(t_fine[::args.sample_ratio],
                data['x_fine'][::args.sample_ratio, 0],
                marker='.',
                markersize=8,
                linestyle='None')
    # Plot observations vs time.  Short time finely sampled
    for i, y_i in enumerate(data['y_fine'].T):
        sampled_observations.plot(t_fine, y_i)
        sampled_observations.plot(t_fine[::args.sample_ratio],
                                  y_i[::args.sample_ratio],
                                  marker='.',
                                  markersize=8,
                                  linestyle='None',
                                  label=f'$y_{i}$')

    # Plot observations vs time.  Long time coarsely sampled
    for i, y_i in enumerate(data['y_coarse'].T):
        observations.plot(t_coarse, y_i, label=f'$y_{i}$')
    # First component of state and filtered estimate.
    filtered.plot(t_coarse, data['x_coarse'][:, 0], label='$x_0$')
    filtered.plot(t_coarse,
                  data['forward_means'][:, 0],
                  label='$\hat x_0$ forward')
    # Error of filter estimate and calculated variance of filter
    plot_error(errors, t_coarse, data['forward_covariances'],
               data['forward_means'][:, 0] - data['x_coarse'][:, 0],
               'filter error')

    # Legends for all axes
    for axis in (phase_portrait, states, sampled_observations, observations,
                 filtered, errors):
        axis.legend()

    # FixMe: Force matching ticks
    #sampled_observations.get_shared_x_axes().join(states, sampled_observations)
    #errors.get_shared_x_axes().join(errors, filtered, observations)
    #errors.get_shared_y_axes().join(errors, filtered)

    # Drop tick labels on some shared axes.  FixMe: Drop some more
    for axis in (states, observations, filtered):
        axis.set_xticklabels([])

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
