r"""max_like.py

Check likelihood of data from the state space model defined in linear_simulatinon.py

"""

import sys
import argparse
import pickle

import numpy
import numpy.random
import scipy.stats

import hmm.state_space
import plotscripts.utilities

import linear_simulation


def parse_args(argv):
    """Parse the command line.
    """

    parser = argparse.ArgumentParser(description='Illustrate MLE.')
    linear_simulation.system_args(parser)

    parser.add_argument('--n_samples',
                        type=int,
                        default=100000,
                        help='Number of samples')
    parser.add_argument('--n_b',
                        type=int,
                        default=10,
                        help='Number of b values')
    parser.add_argument('--b_range',
                        type=float,
                        nargs=2,
                        default=[.75, 1.25],
                        help='Fractional range of b values')
    parser.add_argument('--random_seed', type=int, default=9)
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make a plot of log likelihood vs b

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    rng = numpy.random.default_rng(args.random_seed)

    # Make the nominal model and simulate the data
    d_t = 2 * numpy.pi / (args.omega * args.sample_rate)
    system, initial_dist = linear_simulation.make_system(args, d_t, rng)
    x_data, y_data = system.simulate_n_steps(initial_dist, args.n_samples)

    # Calculate the log likelihood for models with a range of b
    # (system noise) values
    b_array = numpy.linspace(args.b_range[0], args.b_range[1],
                             args.n_b) * args.b
    log_like = numpy.empty(b_array.shape)
    for i, b_i in enumerate(b_array):
        args.b = b_i
        system_b, b_dist = linear_simulation.make_system(args, d_t, rng)
        log_like[i] = system_b.log_likelihood(b_dist, y_data)

    fig, (axis_a, axis_b) = pyplot.subplots(nrows=2, figsize=(6, 8))

    axis_a.plot(x_data[:, 0])
    axis_a.set_xlabel('$t$')
    axis_a.set_ylabel('$x_0$')

    axis_b.plot(b_array, log_like)
    axis_b.set_xlabel('$b$')
    axis_b.set_ylabel('Log Likelihood')

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
