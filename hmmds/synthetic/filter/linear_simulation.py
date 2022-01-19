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


After relaxing, the following equation gives the scale of x

   E(x^2) = \frac{b^2}{1 - e^{-2*a*dt}}

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

    period = 1.0
    omega = (2 * numpy.pi) / period

    parser.add_argument('--sample_rate',
                        type=float,
                        default=10.0,
                        help='number of samples per cycle')
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


def parse_args(argv):
    """Parse the command line.
    """

    parser = argparse.ArgumentParser(
        description=
        'Generate a sequence of observations from a state space model.')
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
    parser.add_argument('--random_seed', type=int, default=3)
    return parser.parse_args(argv)


def make_system(args, dt, rng):
    """Make a system instance
    
    Args:
        args: Command line arguments
        dt: Sample interval

    Returns:
        A system instance
    """
    # pylint: disable = invalid-name
    a = numpy.array([[numpy.cos(args.omega * dt),
                      numpy.sin(args.omega * dt)],
                     [-numpy.sin(args.omega * dt),
                      numpy.cos(args.omega * dt)]]) * numpy.exp(-args.a * dt)
    b = numpy.eye(2) * args.b * numpy.sqrt(dt)  # State noise is b * Normal(0,I)
    c = numpy.array([
        [args.c, 0.0],
    ])
    d = numpy.array([args.d],
                    dtype=numpy.float64)  # Observation noise is c * Normal(0,I)
    sigma_squared = b[0, 0]**2 / (1 - numpy.exp(-2 * args.a * dt))
    stationary_distribution = hmm.state_space.MultivariateNormal(
        numpy.zeros(2),
        numpy.eye(2) * sigma_squared, rng)
    return hmm.state_space.LinearGaussian(a, b, c, d,
                                          rng), stationary_distribution


def main(argv=None):
    """Writes time series to files specified by options --xyzfile,
    --quantfile, and or --TSintro.

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    dt_fine = 2 * numpy.pi / (args.omega * args.sample_rate)
    system_fine, initial_fine = make_system(args, dt_fine, rng)
    dt_coarse = dt_fine * args.sample_ratio
    system_coarse, initial_coarse = make_system(args, dt_coarse, rng)

    x_coarse, y_coarse = system_coarse.simulate_n_steps(initial_coarse,
                                                        args.n_coarse)
    means, covariances = system_coarse.filter(
        initial_coarse, y_coarse)  # Run Kalman filter on simulated observations
    x_fine, y_fine = system_fine.simulate_n_steps(initial_fine, args.n_fine)

    with open(args.data, 'wb') as _file:
        pickle.dump(
            {
                'dt_fine': dt_fine,
                'dt_coarse': dt_coarse,
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
