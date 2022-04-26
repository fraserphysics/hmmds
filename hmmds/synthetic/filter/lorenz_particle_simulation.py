"""lorenz_simulation.py Simulate Lorenz system to make data.  Then
exercise particle filter.

Derived from lorenz_simulation.py
"""
from __future__ import annotations  # Enables, eg, (self: System

import sys
import argparse
import os.path
import pickle
import typing

import numpy
import numpy.random

import hmm.state_space
import hmm.particle
import hmmds.synthetic.filter.lorenz_sde # Cython code

import linear_map_simulation


class LorenzSystem(hmm.particle.System):
    # This class is the essential new element in this file
    def __init__(self: LorenzSystem, dt, s, r, b, state_covariance,
                 observation_map, observation_covariance, initial_mean,
                 initial_covariance, rng):
        """A class derived from hmm.particle.LinearSystem

        Args:
            dt:
            s:
            r:
            b:
            state_covariance:
            observation_map:
            observation_covariance:
            initial_mean:
            initial_covariance:
            rng:

        """
        self.dt = dt
        self.s = s
        self.r = r
        self.b = b
        self.observation_map = observation_map
        self.initial_distribution = hmm.state_space.MultivariateNormal(
            initial_mean, initial_covariance, rng)
        self.rng = rng
        self.y_dimension, self.x_dimension = observation_map.shape
        self.transition_distribution = hmm.state_space.MultivariateNormal(
            numpy.zeros(self.x_dimension), state_covariance, rng)
        self.observation_distribution = hmm.state_space.MultivariateNormal(
            numpy.zeros(self.y_dimension), observation_covariance, rng)

        # Calculate parameters for the importance function
        inverse_observation_covariance = numpy.linalg.inv(
            observation_covariance)
        info_y = numpy.linalg.multi_dot([
            observation_map.T, inverse_observation_covariance, observation_map
        ])
        importance_covariance = numpy.linalg.inv(
            numpy.linalg.inv(state_covariance) + info_y)
        self.importance_distribution = hmm.state_space.MultivariateNormal(
            numpy.zeros(self.x_dimension), importance_covariance, rng)

        self.importance_gain = numpy.linalg.multi_dot([
            importance_covariance, observation_map.T,
            inverse_observation_covariance
        ])

    def transition(self: LorenzSystem, x_next, x_now):
        """Calculate the probability density p(x_next|x_now)
        """
        mean_next = hmmds.synthetic.filter.lorenz_sde.lorenz_integrate(
            x_now, 0.0, self.dt, self.s, self.r, self.b)
        return self.transition_distribution(x_next - mean_next)

    def observation(self: LorenzSystem, y_now, x_now):
        """Calculate the probability density p(y_now|x_now)
        """
        y_mean = numpy.dot(self.observation_map, x_now)
        return self.observation_distribution(y_now - y_mean)

    def importance_0(self: LorenzSystem, y_0):
        x_0 = self.initial_distribution.draw()
        q_value = self.initial_distribution(x_0)
        return x_0, q_value

    def importance(self: LorenzSystem, y_next, x_now):
        """Generate a random x_next and calculate q(x_next|y_next, x_now)
        Args:
            y_next
            x_now

        Return:
            (x_next, q(x_next|y_next, x_now)

        q(x_next|y_next, x_now) = p(x_next|y_next, x_now)

        """
        forecast_mean = hmmds.synthetic.filter.lorenz_sde.lorenz_integrate(
            x_now, 0.0, self.dt, self.s, self.r, self.b)
        forecast_error = y_next - numpy.dot(self.observation_map, forecast_mean)
        update_mean = forecast_mean + numpy.dot(self.importance_gain,
                                                forecast_error)

        noise = self.importance_distribution.draw()
        x_next = update_mean + noise
        q_value = self.importance_distribution(noise)
        return x_next, q_value

    def prior(self: LorenzSystem, x_0):
        return self.initial_distribution(x_0)

# make_system is copied from lorenz_simulation.py  FixMe: call it from there
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
        g = numpy.array([[0, 0, .5], [-2.0, 2.0, 0]])
        return numpy.dot(g, state), g

    x_dim = 3
    state_noise = numpy.eye(x_dim) * args.b
    y_dim = observation_function(0, numpy.ones(x_dim))[0].shape[0]
    observation_noise = numpy.eye(y_dim) * args.d

    system = hmmds.synthetic.filter.lorenz_sde.SDE(dx_dt,
                                                   tangent,
                                                   state_noise,
                                                   observation_function,
                                                   observation_noise,
                                                   dt,
                                                   x_dim,
                                                   ivp_args=(10.0, 28.0,
                                                             8.0 / 3),
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
    under = system_coarse.system
    state_noise_covariance = (args.dt * numpy.dot(
            under.unit_state_noise, under.unit_state_noise.T))
    observation_noise_covariance = numpy.dot(under.observation_noise_multiplier,
                                           under.observation_noise_multiplier.T)
    observation_map = numpy.array([[0, 0, .5], [-2.0, 2.0, 0]])
    s, r, b = (10.0, 28.0, 8.0/3)
    system = LorenzSystem(
        args.dt, s, r, b,
        state_noise_covariance,
        observation_map,
        observation_noise_covariance, initial_coarse.mean,
        initial_coarse.covariance, rng)
    n_times = len(y_coarse)
    n_particles = numpy.ones(n_times, dtype=int) * 100
    n_particles[0:3] *= 10
    particles, forward_means, forward_covariances, log_likelihood = system.forward_filter(
        y_coarse, n_particles, threshold=0.5)
    print(f"log_likelihood: {log_likelihood}")

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
