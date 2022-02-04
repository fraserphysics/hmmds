"""lorenz_simulation.py Simulate Lorenz system to make data.  Then
exercise filter and smooth.

"""
import sys
import typing
import pickle
import argparse

import numpy

import hmm.state_space

import linear_map_simulation


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

    def dx_dt(t, x):
        s, r, b = (10.0, 28.0, 8.0 / 3)
        return numpy.array([
            s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]
        ])

    def tangent(t, x_dx):
        result = numpy.empty(12)  # Allocate storage for result
        s, r, b = (10.0, 28.0, 8.0 / 3)

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
        g = numpy.array([[0, 0, .5], [-2.0, 2.0, 0]])
        return numpy.dot(g, state), g

    x_dim = 3
    state_noise = numpy.ones(x_dim) * args.b
    y_dim = observation_function(0, numpy.ones(x_dim))[0].shape[0]
    observation_noise = numpy.eye(y_dim) * args.d

    system = hmm.state_space.SDE(dx_dt,
                                 tangent,
                                 state_noise,
                                 observation_function,
                                 observation_noise,
                                 dt,
                                 x_dim,
                                 fudge=args.fudge)
    initial_state = system.relax(500)[0]
    final_state, stationary_distribution = system.relax(
        500, initial_state=initial_state)
    result = hmm.state_space.NonStationary(system, dt, rng)
    return result, stationary_distribution, final_state


def more_args(parser: argparse.ArgumentParser):
    """Arguments to add to those from linear_map_simulation.py
    """
    parser.add_argument('--dt',
                        type=float,
                        default=0.15,
                        help='sampling interval')
    parser.add_argument('--fudge',
                        type=float,
                        default=100,
                        help='sampling interval')


def main(argv=None):
    """
    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = linear_map_simulation.parse_args(
        argv, (linear_map_simulation.system_args, more_args))

    rng = numpy.random.default_rng(args.random_seed)

    dt_fine = args.dt / args.sample_ratio
    dt_coarse = args.dt

    system_coarse, initial_coarse, coarse_state = make_system(
        args, dt_coarse, rng)
    system_fine, initial_fine, fine_state = make_system(args, dt_fine, rng)

    x_fine, y_fine = system_fine.simulate_n_steps(initial_fine,
                                                  args.n_fine,
                                                  states_0=coarse_state)
    x_coarse, y_coarse = system_coarse.simulate_n_steps(initial_coarse,
                                                        args.n_coarse,
                                                        states_0=coarse_state)

    forward_means, forward_covariances = system_coarse.forward_filter(
        initial_coarse, y_coarse)
    # information_means, informations = system_coarse.backward_information_filter(
    #     y_coarse)
    # smooth_means, smooth_covariances = system_coarse.smooth(
    #     initial_coarse, y_coarse)

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
                # 'smooth_means': smooth_means,
                # 'smooth_covariances': smooth_covariances,
                # 'information_means': information_means,
                # 'informations': informations,
            },
            _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
