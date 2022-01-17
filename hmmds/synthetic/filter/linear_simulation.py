r"""linear_simulation.py

Generate a sequence of observations from the following state space
model
.. math::
    x_{t+1} = A x_t + B V_n

    y_t = C x_t + D W_n

    A = [cos(\omega * dt) sin(\omega * dt)] * \exp(-a dt)
        [-sin(\omega * dt) cos(\omega * dt)]

    B = [b 0]
        [0 b]

    C = [c 0]

V and W are unit variance iid Gaussian noise with dimension 2 and 1
respectively and the parameters \omega, dt, a, b, c, and d are
arguments to the module.

"""

import sys
import argparse
import os.path
import pickle

import numpy
import numpy.random

import hmm.state_space


def parse_args(argv):
    """Parse the command line.
    """
    period = 1.0
    omega = (2 * numpy.pi) / period

    parser = argparse.ArgumentParser(
        description=
        'Generate a sequence of observations from a state space model.')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./',
                        help='Writes data to this directory')
    parser.add_argument('--sample_rate',
                        type=float,
                        default=10.0,
                        help='number of samples per cycle')
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
    parser.add_argument('--mean',
                        type=float,
                        nargs=2,
                        default=[0, 0],
                        help='Initial mean')
    parser.add_argument(
        '--covariance',
        type=float,
        nargs=3,
        default=[.25, 0, .25],
        help='Initial covariance components (1,1), (1,2), and (2,2)')
    parser.add_argument('--omega',
                        type=float,
                        default=omega,
                        help='system rotation rate')
    parser.add_argument('--a',
                        type=float,
                        default=0.001 * omega,
                        help='system dissipation rate')
    parser.add_argument('--b',
                        type=float,
                        default=.01,
                        help='System noise multiplier')
    parser.add_argument('--c', type=float, default=0.5, help='Observation map')
    parser.add_argument('--d',
                        type=float,
                        default=0.2,
                        help='Observation noise multiplier')
    parser.add_argument('data', type=str, help='Path to store data')
    parser.add_argument('--random_seed', type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None):
    """Writes time series to files specified by options --xyzfile,
    --quantfile, and or --TSintro.

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    def make_system(args, dt):
        """Make a system instance
        
        Args:
            args: Command line arguments
            dt: Sample interval

        Returns:
            A system instance
        """
        # pylint: disable = invalid-name
        a = numpy.array([[
            numpy.cos(args.omega * dt),
            numpy.sin(args.omega * dt)
        ], [-numpy.sin(args.omega * dt),
            numpy.cos(args.omega * dt)]]) * numpy.exp(-args.a * dt)
        b = numpy.eye(2) * args.b * numpy.sqrt(
            dt)  # State noise is b * Normal(0,I)
        c = numpy.array([
            [args.c, 0.0],
        ])
        d = numpy.array(
            [args.d],
            dtype=numpy.float64)  # Observation noise is c * Normal(0,I)
        return hmm.state_space.LinearGaussian(a, b, c, d, rng)

    mean = numpy.array(args.mean)
    covariance = numpy.array([[args.covariance[0], args.covariance[1]],
                              [args.covariance[1], args.covariance[0]]])

    initial_dist = hmm.state_space.MultivariateNormal(mean, covariance, rng)
    initial_estimate = hmm.state_space.MultivariateNormal(mean, covariance, rng)

    system_fine = make_system(args,
                              2 * numpy.pi / (args.omega * args.sample_rate))
    system_coarse = make_system(
        args,
        2 * numpy.pi * args.sample_ratio / (args.omega * args.sample_rate))

    x_coarse, y_coarse = system_coarse.simulate_n_steps(initial_dist,
                                                        args.n_coarse)
    means, covariances = system_coarse.filter(
        initial_estimate,
        y_coarse)  # Run Kalman filter on simulated observations
    x_fine, y_fine = system_fine.simulate_n_steps(initial_dist, args.n_fine)

    with open(args.data, 'wb') as _file:
        pickle.dump(
            {
                'x_fine': x_fine,
                'y_fine': y_fine,
                'x_coarse': x_coarse,
                'y_coarse': y_coarse,
                'means': means,
                'covariances': covariances
            }, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
