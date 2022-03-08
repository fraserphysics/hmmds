"""linear_sde_simulation.py Exercise SDE class with simple linear ODE.

"""
import sys

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
    SDE methods on an ODE that is easy to integrate.

    """

    # State dynamics d/dt x(t) = a * x
    a = numpy.array([[-.01, .2], [-.2, -.01]])
    # Observation y = c * x
    c = numpy.array([[0, 0.5]])

    observation_noise = numpy.eye(1) * args.d

    # The next three functions are passed to SDE.__init__
    def observation_function(t, x):
        return numpy.dot(c, x), c

    def dx_dt(t, x, a):
        return numpy.dot(a, x)

    def tangent(t, x_d, a):
        dim_x_d = 6  # 2 for x 4 for d_x
        assert x_d.shape == (dim_x_d,)
        x = x_d[:2]
        derivative = x_d[2:].reshape((2, 2))
        result = numpy.empty(6)
        result[:2] = numpy.dot(a, x)
        result[2:] = numpy.dot(a, derivative).reshape(-1)
        return result

    state_noise = numpy.ones(2) * args.b
    dt = 2 * numpy.pi / 10  # 10 samples per cycle
    x_dim = 2
    system = hmm.state_space.SDE(dx_dt,
                                 tangent,
                                 state_noise,
                                 observation_function,
                                 observation_noise,
                                 dt,
                                 x_dim,
                                 ivp_args=(a,))
    initial_state = system.relax(500)[0]
    stationary_distribution = system.relax(500, initial_state=initial_state)[1]
    result = hmm.state_space.NonStationary(system, dt, rng)
    return result, stationary_distribution


def main():
    """
    """
    return linear_map_simulation.main(sys.argv[1:], make_system=make_system)


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
