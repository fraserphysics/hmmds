"""ekf.py Extended Kalman Filter for laser data.

Derived from hmmds.synthetic.filter.lorenz_simulation

"""
import sys
import typing
import pickle
import argparse

import numpy

import hmm.state_space
import hmmds.synthetic.filter.lorenz_sde

import explore


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Simulate and filter laser data')
    parser.add_argument('--state_noise',
                        type=float,
                        default=0.01,
                        help='State noise amplitude')
    parser.add_argument('--observation_noise',
                        type=float,
                        default=2.0,
                        help='Observation noise amplitude')
    parser.add_argument('--s',
                        type=float,
                        default=10.0,
                        help='Lorenz parameter')
    parser.add_argument('--r',
                        type=float,
                        default=30.0,
                        help='Lorenz parameter')
    parser.add_argument('--b',
                        type=float,
                        default=8.0 / 3,
                        help='Lorenz parameter')
    parser.add_argument('--dt',
                        type=float,
                        default=0.04,
                        help='sampling interval')
    parser.add_argument('--delta_x',
                        type=float,
                        default=2.993,
                        help='Start wrt fixed point')
    parser.add_argument('--x_ratio',
                        type=float,
                        default=0.742424,
                        help='observation multiplier')
    parser.add_argument('--offset',
                        type=int,
                        default=15,
                        help='added to observations')
    parser.add_argument('--fudge',
                        type=float,
                        default=300,
                        help='Multiply state noise scale for filtering')
    parser.add_argument('--n_samples',
                        type=int,
                        default=1000,
                        help='Number of samples')
    parser.add_argument('--data', type=str, default='test_ekf', help='Path to store data')
    parser.add_argument('--random_seed', type=int, default=9)
    return parser.parse_args(argv)


def make_system(args, dt, rng):
    """Make an SDE system instance

    Args:
        args: Command line arguments
        dt: Sample interval
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
        y_value = int(args.x_ratio * state[0]**2 + args.offset)
        y = numpy.array([y_value])
        dy_dx = numpy.array([[args.x_ratio * 2 * state[0], 0, 0]])
        return y, dy_dx

    fixed_point = explore.FixedPoint(args.r)
    x_dim = 3
    state_noise = numpy.ones(x_dim) * args.state_noise
    y_dim = 1
    observation_noise = numpy.eye(y_dim) * args.observation_noise

    # lorenz_sde.SDE only uses Cython for methods forecast and simulate
    # FixMe: Lorenz parameters are built into lorenz_sde.SDE
    system = hmmds.synthetic.filter.lorenz_sde.SDE(dx_dt,
                                                   tangent,
                                                   state_noise,
                                                   observation_function,
                                                   observation_noise,
                                                   dt,
                                                   x_dim,
                                                   ivp_args=(args.s,args.r,args.b),
                                                   fudge=args.fudge)
    initial_mean = fixed_point.initial_state(args.delta_x)
    initial_covariance = numpy.outer(state_noise, state_noise)
    initial_distribution = hmm.state_space.MultivariateNormal(
        initial_mean, initial_covariance)
    result = hmm.state_space.NonStationary(system, dt, rng)
    return result, initial_distribution, initial_mean


def main(argv=None):
    """
    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    dt = args.dt

    system, initial_distribution, initial_state = make_system(
        args, args.dt, rng)
    states, observations = system.simulate_n_steps(initial_distribution,
                                                   args.n_samples,
                                                   states_0=initial_state)

    forward_means, forward_covariances = system.forward_filter(
        initial_distribution, observations)

    with open(args.data, 'wb') as _file:
        pickle.dump(
            {
                'dt': dt,
                'states': states,
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
