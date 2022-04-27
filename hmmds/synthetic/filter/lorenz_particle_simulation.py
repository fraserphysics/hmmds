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
import hmmds.synthetic.filter.lorenz_sde  # Cython code
import hmmds.synthetic.filter.linear_map_simulation
import hmmds.synthetic.filter.lorenz_simulation


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


def main(argv=None):
    """Imitates and invokes code from lorenz_simulation.py to make Lorenz
    data for a plot.  Differs from lorenz_simulation in applying a
    particle filter to data rather than an extended Kalman filter.

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    # Parse same arguments as lorenz_simulation.py
    args = hmmds.synthetic.filter.linear_map_simulation.parse_args(
        argv, (hmmds.synthetic.filter.linear_map_simulation.system_args,
               hmmds.synthetic.filter.lorenz_simulation.more_args))

    # Make Lorenz data
    rng = numpy.random.default_rng(args.random_seed)

    dt_fine = args.dt / args.sample_ratio
    dt_coarse = args.dt

    system_coarse, initial_coarse, coarse_state = hmmds.synthetic.filter.lorenz_simulation.make_system(
        args, dt_coarse, rng)
    system_fine, initial_fine, fine_state = hmmds.synthetic.filter.lorenz_simulation.make_system(
        args, dt_fine, rng)

    x_fine, y_fine = system_fine.simulate_n_steps(initial_fine,
                                                  args.n_fine,
                                                  states_0=coarse_state)
    x_coarse, y_coarse = system_coarse.simulate_n_steps(initial_coarse,
                                                        args.n_coarse,
                                                        states_0=coarse_state)
    # Get parameters for particle filter
    s, r, b = (10.0, 28.0, 8.0 / 3)
    under = system_coarse.system
    state_noise_covariance = (
        args.dt * numpy.dot(under.unit_state_noise, under.unit_state_noise.T))
    fudge = 10  # FixMe: Why?
    observation_noise_covariance = numpy.dot(
        under.observation_noise_multiplier,
        under.observation_noise_multiplier.T) * fudge
    observation_map = numpy.array([[0, 0, .5], [-2.0, 2.0, 0]])
    system = LorenzSystem(args.dt, s, r, b, state_noise_covariance,
                          observation_map, observation_noise_covariance,
                          initial_coarse.mean, initial_coarse.covariance, rng)
    n_times = len(y_coarse)
    n_particles = numpy.ones(n_times, dtype=int) * 300
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
