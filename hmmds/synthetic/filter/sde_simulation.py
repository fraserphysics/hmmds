"""mimic.py Make linear_simulation.py exercise all methods of EKF.

"""
import sys

import numpy

import hmm.state_space

import linear_simulation


def make_system(args, dt, rng):
    """Make an SDE system instance
    
    Args:
        args: Command line arguments
        dt: Sample interval
        rng:

    Returns:
        (An SDE instance, an initial state, an inital distribution)

    The goal is to get linear_simulation.main to exercise all of the
    SDE methods on an ODE that is easy to integrate.

    """

    # State dynamics d/dt x(t) = a * x
    a = numpy.array([[-.01, 1], [-1, -.01]])
    # Observation y = c * x
    c = numpy.array([
        [0, .5],
        [0.01, 0]
    ])

    observation_noise = numpy.eye(2)*0.1

    def observation_function(t, x):
        return numpy.dot(c, x), c

    def dx_dt(t, x):
        return numpy.dot(a, x)

    def tangent(t, x_d):
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
    system = hmm.state_space.SDE(dx_dt, tangent, state_noise, observation_function,
                 observation_noise, dt, x_dim)
    initial_state = system.relax(500)[0]
    stationary_distribution = system.relax(500, initial_state=initial_state)[1]
    result = hmm.state_space.NonStationary(
        system, dt, rng)
    return result, stationary_distribution


def main():
    """
    """
    args = linear_simulation.parse_args(sys.argv[1:],
                                        (linear_simulation.system_args,))
    return linear_simulation.main(sys.argv[1:], make_system=make_system)


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
