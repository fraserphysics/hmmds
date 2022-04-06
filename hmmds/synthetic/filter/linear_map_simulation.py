r"""linear_map_simulation.py

Using a LinearGaussian state space model make the following data for plotting:

1. Sequences of states and observation with a fine sampling interval

2. Sequences of states and observation with a coarse sampling interval

3. Means and covariances from Kalman filtering the coarse observations

4. Means and covariances from backward filtering the coarse observations

5. Means and covariances from smoothing the coarse observations

Here is a description of the LinearGaussian system:

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


def parse_args(argv, additional_args):
    """Parse the command line.
    """

    parser = argparse.ArgumentParser(
        description='Generate data for a state space model figure.')
    for more in additional_args:
        more(parser)
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


def make_linear_stationary(args, dt, rng):
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
    d = numpy.array([[args.d]],
                    dtype=numpy.float64)  # Observation noise is c * Normal(0,I)
    # Calculate stationary distribution
    sigma_squared = b[0, 0]**2 / (1 - numpy.exp(-2 * args.a * dt))
    stationary_distribution = hmm.state_space.MultivariateNormal(
        numpy.zeros(2),
        numpy.eye(2) * sigma_squared, rng)
    return hmm.state_space.LinearStationary(a, b, c, d,
                                            rng), stationary_distribution


def main(argv=None,
         make_system=make_linear_stationary,
         additional_args=(system_args,)):
    """Writes data for plotting to file specified by args.data

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv, additional_args)

    rng = numpy.random.default_rng(args.random_seed)

    dt_fine = 2 * numpy.pi / (args.omega * args.sample_rate)
    dt_coarse = dt_fine * args.sample_ratio

    system_fine, initial_fine = make_system(args, dt_fine, rng)
    system_coarse, initial_coarse = make_system(args, dt_coarse, rng)

    x_fine, y_fine = system_fine.simulate_n_steps(initial_fine, args.n_fine)
    x_coarse, y_coarse = system_coarse.simulate_n_steps(initial_coarse,
                                                        args.n_coarse)

    forward_means, forward_covariances = system_coarse.forward_filter(
        initial_coarse, y_coarse)
    backward_means, backward_informations = system_coarse.backward_information_filter(
        y_coarse, forward_means[-1])
    smooth_means, smooth_covariances = system_coarse.smooth(
        initial_coarse, y_coarse)

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
                'smooth_means': smooth_means,
                'smooth_covariances': smooth_covariances,
                'backward_means': backward_means,
                'backward_informations': backward_informations,
            }, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
