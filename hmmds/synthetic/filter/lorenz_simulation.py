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

import hmm.examples.ekf


def system_args(parser: argparse.ArgumentParser):
    """I separated these so that other modules can import.
    """

    parser.add_argument('--coarse_dt',
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
    parser.add_argument('--fudge',
                        type=float,
                        default=1.0,
                        help='Multiplier of state noise for filtering')


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


def make_system(args, dt, rng):
    """Make a system instance
    
    Args:
        args: Command line arguments
        dt: Sample interval
        rng:

    Returns:
        (A system instance, an initial state, an inital distribution)
    """
    lorenz = hmm.examples.ekf.Lorenz(dt=dt,
                                     state_noise=args.state_noise,
                                     observation_noise=args.observation_noise,
                                     rng=rng,
                                     fudge=args.fudge)
    relax = hmm.examples.ekf.Lorenz(dt=1,
                                    state_noise=args.state_noise,
                                    observation_noise=args.observation_noise,
                                    rng=rng,
                                    fudge=args.fudge)
    n_relax = 1000
    initial_distribution = hmm.state_space.MultivariateNormal(
        numpy.ones(3),
        numpy.eye(3) * .01)
    states, observations = relax.simulate_n_steps(initial_distribution, n_relax)
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
    return hmm.state_space.EKF(lorenz, dt,
                               rng), initial_state, initial_distribution


def main(argv=None):
    """Writes time series to files specified by options --xyzfile,
    --quantfile, and or --TSintro.

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    dt_coarse = args.coarse_dt
    dt_fine = dt_coarse / args.sample_ratio

    ekf_fine, fine_state, fine_distribution = make_system(args, dt_fine, rng)
    ekf_coarse, coarse_state, coarse_distribution = make_system(
        args, dt_coarse, rng)

    def distribution(state):
        return hmm.state_space.MultivariateNormal(state, numpy.eye(3) * 1e-8)

    x_fine, y_fine = ekf_fine.system.simulate_n_steps(distribution(fine_state),
                                                      args.n_fine)
    x_coarse, y_coarse = ekf_coarse.system.simulate_n_steps(
        distribution(coarse_state), args.n_coarse)

    forward_means, forward_covariances = ekf_coarse.forward_filter(
        coarse_distribution, y_coarse)

    with open(args.data, 'wb') as _file:
        pickle.dump(
            {
                'dt_fine': dt_fine,
                'dt_coarse': dt_coarse,
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
