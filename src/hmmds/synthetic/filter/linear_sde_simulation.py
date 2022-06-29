"""linear_sde_simulation.py Exercise SDE class with simple linear ODE.

"""
import sys

import numpy

import hmm.state_space

from hmmds.synthetic.filter import linear_map_simulation


def make_system(args, dt, rng):  # pylint: disable = invalid-name
    """Make an SDE system instance

    Args:
        args: Command line arguments
        dt: Sample interval
        rng:

    Returns:
        (An SDE instance, an initial state, an inital distribution)

    The goal is to get linear_map_simulation.main to exercise all of the
    SDE methods on an ODE that is easy to integrate.

    """

    # State dynamics d/dt x(t) = state_map * x
    state_map = numpy.array([[-.01, .2], [-.2, -.01]])
    # Observation y = observation_map * x
    observation_map = numpy.array([[0, 0.5]])

    observation_noise = numpy.eye(1) * args.d

    # The next three functions are passed to SDE.__init__
    def observation_function(_, x):
        return numpy.dot(observation_map, x), observation_map

    def dx_dt(_, x, state_map):
        return numpy.dot(state_map, x)

    def tangent(_, x_d, state_map):
        dim_x_d = 6  # 2 for x 4 for d_x
        assert x_d.shape == (dim_x_d,)
        x = x_d[:2]
        derivative = x_d[2:].reshape((2, 2))
        result = numpy.empty(6)
        result[:2] = numpy.dot(state_map, x)
        result[2:] = numpy.dot(state_map, derivative).reshape(-1)
        return result

    state_noise = numpy.eye(2) * args.b
    dt = 2 * numpy.pi / 10  # 10 samples per cycle
    x_dim = 2
    system = hmm.state_space.SDE(dx_dt,
                                 tangent,
                                 state_noise,
                                 observation_function,
                                 observation_noise,
                                 dt,
                                 x_dim,
                                 ivp_args=(state_map,))
    initial_state = system.relax(500)[0]
    stationary_distribution = system.relax(500, initial_state=initial_state)[1]
    result = hmm.state_space.NonStationary(system, dt, rng)
    return result, stationary_distribution


def main(argv=None):
    """ Stub to enable calling from testing code.
    """

    if argv is None:
        argv = sys.argv[1:]
    return linear_map_simulation.main(argv, make_system=make_system)


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
