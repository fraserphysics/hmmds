r"""distibution.py

Characterize the distribution of states for the state space model defined in linear_simulatinon.py

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

    parser = argparse.ArgumentParser(
        description=
        'Characterize the distribution of states of a stochastic system.')
    linear_simulation.system_args(parser)

    parser.add_argument('--n_samples',
                        type=int,
                        default=1000,
                        help='Number of samples')
    parser.add_argument('--random_seed', type=int, default=3)
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data', type=str, help='Path to data')
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Writes time series to files specified by options --xyzfile,
    --quantfile, and or --TSintro.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    rng = numpy.random.default_rng(args.random_seed)

    dt = 2 * numpy.pi / (args.omega * args.sample_rate)
    system, initial_dist = linear_simulation.make_system(args, dt, rng)
    std_deviation = numpy.sqrt(initial_dist.covariance[0, 0])

    x_01, _ = system.simulate_n_steps(initial_dist, args.n_samples)
    x_0 = x_01[:, 0]

    with open(args.data, 'wb') as _file:
        pickle.dump({
            'x_0': x_0,
            'std_deviation': std_deviation,
            'dt': dt,
        }, _file)

    # The following will be in plotscripts:

    with open(args.data, 'rb') as _file:
        data = pickle.load(_file)

    quantiles, sorted_x_0 = scipy.stats.probplot(
        data['x_0'],
        dist='norm',
        sparams=(0.0, data['std_deviation']),
        fit=False)

    fig, (axis_a, axis_b) = pyplot.subplots(nrows=2, figsize=(6, 8))

    axis_a.plot(numpy.array(range(len(data['x_0']))) * data['dt'],
                data['x_0'],
                marker='.',
                linestyle='None')
    axis_a.set_xlabel('t')
    axis_a.set_ylabel('$x_0$')

    axis_b.plot(quantiles, sorted_x_0, marker='.', linestyle='None')
    ends = [quantiles[0], quantiles[-1]]
    axis_b.plot(ends, ends)
    axis_b.set_xlabel('Theoretical quantiles')
    axis_b.set_ylabel('$x_0$ quantiles')
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
