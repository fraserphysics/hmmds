"""plot_toy_h.py Make figure that illustrates cross entropy of
extended Kalman filters.  Two plots: (1) -^h vs measurement noise and
# sample time; (2) -^h vs sample time

"""

import sys
import argparse
import pickle

import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Figure to illustrate extended Kalman filter')
    parser.add_argument('--azim',
                        type=float,
                        default=45.0,
                        help='Viewing angle')
    parser.add_argument('--elev',
                        type=float,
                        default=25.0,
                        help='Viewing angle')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data', type=str, help='Path to data')
    parser.add_argument('toy_h', type=str, help='path for result')
    return parser.parse_args(argv)


def main(argv=None):
    """Make figure that illustrates cross entropy of extended Kalman
filters.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    figure = pyplot.figure(figsize=(9, 5))

    with open(args.data, 'rb') as file_:
        data = pickle.load(file_)
    assert set(data.keys()) == set(
        'intercept slope lower upper cross_entropy'.split())

    # Unpack entropy data
    entropy_dict = data['cross_entropy']
    t_sample = numpy.array(sorted(entropy_dict.keys()))
    log_noises = numpy.array(sorted(entropy_dict[t_sample[0]].keys()))
    entropy = numpy.array([
        [entropy_dict[time][noise] for noise in log_noises] for time in t_sample
    ])

    # Make a surface plot of the entropy
    time_grid, noise_grid = numpy.meshgrid(t_sample, log_noises)
    axis_0 = figure.add_subplot(1,
                                2,
                                1,
                                projection='3d',
                                azim=args.azim,
                                elev=args.elev)
    axis_0.set_ylabel(r'$\tau_s$')
    axis_0.set_xlabel(r'$\log_{10}(\tilde \sigma_\epsilon)$')
    axis_0.set_zlabel(r'$-\hat h$')
    axis_0.plot_surface(
        noise_grid,
        time_grid,
        entropy.T,
        rstride=1,
        cstride=1,
        cmap=pyplot.cm.hsv,  # pylint: disable=no-member
        linewidth=1)

    # Plot a line with slope that matches Lyapunov exponent and a line from the ridge
    axis_1 = figure.add_subplot(1, 2, 2)
    axis_1.plot(t_sample,
                data['lower'],
                'rd',
                label=r'$\sigma_\epsilon=10^{-4}$')
    axis_1.plot(t_sample, data['upper'], 'go', label=r'ridge')
    y = data['intercept'] + t_sample * data['slope']
    axis_1.plot(t_sample, y, 'b', label=r'theory')
    axis_1.legend()
    axis_1.set_ylabel(r'$-\hat h$')
    axis_1.set_xlabel(r'$\tau_s$')

    if args.show:
        pyplot.show()
    figure.savefig(args.toy_h)

    return 0


if __name__ == "__main__":
    sys.exit(main())
