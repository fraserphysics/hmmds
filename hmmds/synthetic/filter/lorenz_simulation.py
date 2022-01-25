r"""lorenz_simulation.py Make data for Extended Kalman Filtering plots

Imitate linear_simulation.py and make the following data:

1. Sequences of states and observation with a fine sampling interval

2. Sequences of states and observation with a coarse sampling interval

3. Means and covariances from Kalman filtering the coarse observations

4. Means and covariances from backward filtering the coarse observations

5. Means and covariances from smoothing the coarse observations

"""

import sys
import argparse
import os.path
import pickle

import numpy
import numpy.random

import hmm.state_space


def system_args(parser: argparse.ArgumentParser):
    """I separated these so that other modules can import.
    """

    parser.add_argument('--coarse_d_t',
                        type=float,
                        default=0.15,
                        help='Sample interval')
    parser.add_argument('--state_noise',
                        type=float,
                        default=.001,
                        help='Std deviation of system noise')
    parser.add_argument('--observation_noise',
                        type=float,
                        default=0.01,
                        help='Std deviation of observation noise')


def parse_args(argv):
    """Parse the command line.
    """

    parser = argparse.ArgumentParser(
        description='Generate Lorenz data for an EKF figure.')
    system_args(parser)
    parser.add_argument('--sample_ratio',
                        type=int,
                        default=10,
                        help='Number of fine samples per coarse sample')
    parser.add_argument('--n_fine',
                        type=int,
                        default=1000,
                        help='Number of fine samples')
    parser.add_argument('--n_coarse',
                        type=int,
                        default=1000,
                        help='Number of coarse samples')
    parser.add_argument('data', type=str, help='Path to store data')
    parser.add_argument('--random_seed', type=int, default=9)
    return parser.parse_args(argv)


def make_system(args, d_t, rng):
    """Make a system instance
    
    Args:
        args: Command line arguments
        d_t: Sample interval
        rng:

    Returns:
        (A system instance, an initial state, an inital distribution)
    """
    lorenz = hmm.state_space.Lorenz(d_t=d_t,
                                    state_noise=args.state_noise,
                                    observation_noise=args.observation_noise,
                                    rng=rng)
    n_relax = 1000
    initial_time = 0.0
    states, observations = lorenz.simulate_n_steps(numpy.ones(3), initial_time,
                                                   n_relax)
    assert states.shape == (n_relax, 3)
    mean = numpy.sum(states, axis=0) / n_relax
    assert mean.shape == (3,)
    diffs = states - mean
    assert diffs.shape == (n_relax, 3)
    covariance = numpy.dot(diffs.T, diffs) / n_relax
    assert covariance.shape == (3, 3)
    initial_state = states[-1]
    initial_distribution = hmm.state_space.MultivariateNormal(
        mean, covariance, rng)
    return hmm.state_space.EKF(lorenz, d_t,
                               rng), initial_state, initial_distribution


def main(argv=None):
    """Writes time series to files specified by options --xyzfile,
    --quantfile, and or --TSintro.

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    d_t_coarse = args.coarse_d_t
    d_t_fine = d_t_coarse / args.sample_ratio

    ekf_fine, fine_state, fine_distribution = make_system(args, d_t_fine, rng)
    ekf_coarse, coarse_state, coarse_distribution = make_system(
        args, d_t_coarse, rng)

    t_initial = 0.0  # A place holder
    x_fine, y_fine = ekf_fine.system.simulate_n_steps(fine_state, t_initial,
                                                      args.n_fine)
    x_coarse, y_coarse = ekf_coarse.system.simulate_n_steps(
        coarse_state, t_initial, args.n_coarse)

    forward_means, forward_covariances = ekf_coarse.forward_filter(
        coarse_distribution, y_coarse)

    with open(args.data, 'wb') as _file:
        pickle.dump(
            {
                'dt_fine': d_t_fine,
                'dt_coarse': d_t_coarse,
                'x_fine': x_fine,
                'y_fine': y_fine,
                'x_coarse': x_coarse,
                'y_coarse': y_coarse,
                'forward_means': forward_means,
                'forward_covariances': forward_covariances,
            }, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
