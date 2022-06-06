"""ekf.py Apply extended Kalman Filter to laser data.

Derived from hmmds.synthetic.filter.lorenz_simulation

"""
import sys
import typing
import pickle
import argparse

import numpy

import hmm.state_space
import hmmds.synthetic.filter.lorenz_sde

import optimize
import plotscripts.introduction.laser


def parse_args(argv):
    """Define parser and parse command line.  This code fetches many
    arguments and defalut values from optimize.py

    """

    # Get several arguments and default values from optimize.py
    parameters = optimize.Parameters()
    parser = argparse.ArgumentParser(
        description='Simulate and filter laser data')
    for key in parameters.variables + 'laser_dt '.split():
        parser.add_argument(f'--{key}',
                            type=float,
                            default=getattr(parameters, key))
    parser.add_argument('--fudge',
                        type=float,
                        default=1.0,
                        help='Multiply state noise scale for filtering')
    parser.add_argument('--LaserData',
                        type=str,
                        default='LP5.DAT',
                        help='Path to laser data')
    parser.add_argument('--result',
                        type=str,
                        default='test_ekf',
                        help='Path to store data')
    parser.add_argument('--random_seed', type=int, default=9)
    return parser.parse_args(argv)


def make_non_stationary(args, rng):
    """Make an SDE system instance

    Args:
        args: Command line arguments
        rng:

    Returns:
        (An SDE instance, an initial state, an inital distribution)

    The goal is to get linear_map_simulation.main to exercise all of the
    SDE methods on the Lorenz system.

    """

    # The next three functions are passed to SDE.__init__

    def dx_dt(t, x, s, r, b):
        return numpy.array([
            s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]
        ])

    def tangent(t, x_dx, s, r, b):
        result = numpy.empty(12)  # Allocate storage for result

        # Unpack state and derivative from argument
        x = x_dx[:3]
        dx_dx0 = x_dx[3:].reshape((3, 3))

        # First three components are the value of the vector field F(x)
        result[:3] = dx_dt(t, x)

        dF = numpy.array([  # The derivative of F wrt x
            [-s, s, 0], [r - x[2], -1, -x[0]], [x[1], x[0], -b]
        ])

        # Assign the tangent part of the return value.
        result[3:] = numpy.dot(dF, dx_dx0).reshape(-1)

        return result

    def observation_function(
            t, state) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """Calculate observation and its derivative
        """
        y = numpy.array([args.x_ratio * state[0]**2 + args.offset])
        dy_dx = numpy.array([[args.x_ratio * 2 * state[0], 0, 0]])
        return y, dy_dx

    x_dim = 3
    state_noise = numpy.ones(x_dim) * args.state_noise
    y_dim = 1
    observation_noise = numpy.eye(y_dim) * args.observation_noise

    # lorenz_sde.SDE only uses Cython for methods forecast and simulate
    system = hmmds.synthetic.filter.lorenz_sde.SDE(dx_dt,
                                                   tangent,
                                                   state_noise,
                                                   observation_function,
                                                   observation_noise,
                                                   args.laser_dt,
                                                   x_dim,
                                                   ivp_args=(args.s, args.r,
                                                             args.b),
                                                   fudge=args.fudge)
    initial_mean = numpy.array(
        [args.x_initial_0, args.x_initial_1, args.x_initial_2])
    initial_covariance = numpy.outer(state_noise, state_noise)
    initial_distribution = hmm.state_space.MultivariateNormal(
        initial_mean, initial_covariance)
    result = hmm.state_space.NonStationary(system, args.laser_dt, rng)
    return result, initial_distribution, initial_mean


def main(argv=None):
    """
    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    non_stationary, initial_distribution, initial_state = make_non_stationary(
        args, rng)
    laser_data = plotscripts.introduction.laser.read_data(args.LaserData)
    assert laser_data.shape == (2, 2876)
    observations = laser_data[1, :].astype(int).reshape((2876, 1))
    forward_means, forward_covariances = non_stationary.forward_filter(
        initial_distribution, observations)
    log_likelihood = non_stationary.log_likelihood(initial_distribution,
                                                   observations)
    print(f'log_likelihood={log_likelihood}')

    with open(args.result, 'wb') as _file:
        pickle.dump(
            {
                'dt': args.laser_dt,
                'observations': observations,
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
