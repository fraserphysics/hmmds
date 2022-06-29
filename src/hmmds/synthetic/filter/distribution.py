r"""distibution.py: Compare distribution of simulated data to formula.

linear_map_simulation.py simulates a 2-d linear sde.  It has code that
calculates parameters of a multivariate normal for the stationary
distribution.  This module prepares for a probabilty plot by writing
simulated data and the result of variance expected from the formula in
linear_map_simulation.py.

python distribution.py --n_samples 10000 --a 0.05 --b 0.2 output_data

"""

import argparse
import pickle
import sys

import numpy
import numpy.random

from hmmds.synthetic.filter import linear_map_simulation


def parse_args(argv):
    """Parse the command line.
    """

    parser = argparse.ArgumentParser(
        description=
        'Characterize the distribution of states of a stochastic system.')

    # Get arguments from linear_map_simulation.
    linear_map_simulation.system_args(parser)

    parser.add_argument('--n_samples',
                        type=int,
                        default=1000,
                        help='Number of samples')
    parser.add_argument('--random_seed', type=int, default=3)
    parser.add_argument('data', type=str, help='Path to data')
    return parser.parse_args(argv)


def main(argv=None):
    """Writes simulated data and calculated distribution parameters to
       pickle file.

    """

    # Like log_likelihood, pylint: disable = duplicate-code
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    rng = numpy.random.default_rng(args.random_seed)
    # dt is OK pylint: disable = invalid-name
    dt = 2 * numpy.pi / (args.omega * args.sample_rate)
    system, initial_dist = linear_map_simulation.make_linear_stationary(
        args, dt, rng)
    std_deviation = numpy.sqrt(initial_dist.covariance[0, 0])

    x_01, _ = system.simulate_n_steps(initial_dist, args.n_samples)
    x_0 = x_01[:, 0]

    with open(args.data, 'wb') as _file:
        pickle.dump({
            'x_0': x_0,
            'std_deviation': std_deviation,
            'dt': dt,
        }, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
