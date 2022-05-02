"""particle.py Apply particle filter to laser data.


"""
from __future__ import annotations

import sys
import typing
import pickle
import argparse

import numpy

import hmm.state_space
import hmm.particle
import hmmds.synthetic.filter.lorenz_sde

import optimize
import plotscripts.introduction.laser


class LorenzSystem(hmm.particle.System):
    # From hmmds/synthetic/filter/lorenz_particle_simulation.py
    def __init__(self: LorenzSystem, dt, s, r, b, state_covariance, x_ratio,
                 offset, observation_covariance, initial_mean,
                 initial_covariance, rng):
        """A class derived from hmm.particle.LinearSystem

        Args:
            dt:
            s:
            r:
            b:
            state_covariance:
            x_ratio:
            offset:
            observation_covariance:
            initial_mean:
            initial_covariance:
            rng:

        """
        self.dt = dt
        self.s = s
        self.r = r
        self.b = b
        self.x_ratio = x_ratio
        self.offset = offset
        self.initial_distribution = hmm.state_space.MultivariateNormal(
            initial_mean, initial_covariance, rng)
        self.rng = rng
        self.y_dimension, self.x_dimension = (1, 3)
        self.transition_distribution = hmm.state_space.MultivariateNormal(
            numpy.zeros(self.x_dimension), state_covariance, rng)
        self.observation_distribution = hmm.state_space.MultivariateNormal(
            numpy.zeros(self.y_dimension), observation_covariance, rng)

        # Calculate parameters for the importance function
        self.inverse_observation_covariance = numpy.linalg.inv(
            observation_covariance)
        self.inverse_state_covariance = numpy.linalg.inv(state_covariance)

    def transition(self: LorenzSystem, x_next, x_now):
        """Calculate the probability density p(x_next|x_now)
        """
        mean_next = hmmds.synthetic.filter.lorenz_sde.lorenz_integrate(
            x_now, 0.0, self.dt, self.s, self.r, self.b)
        return self.transition_distribution(x_next - mean_next)

    def observation_map(self: LorenzSystem, state):
        """Calculate mean and derivative of observation function
        Args:
            state:

        O(x) = r*(x_0)^2 + o
        O'(x) = 2*r*x_0

        """
        ratio = self.x_ratio
        x_0 = state[0]
        value = numpy.array([ratio * x_0 * x_0 + self.offset])
        derivative = numpy.array([[2 * ratio * x_0, 0.0, 0.0]])
        return value, derivative

    def observation(self: LorenzSystem, y_now, x_now):
        """Calculate the probability density p(y_now|x_now)
        """
        mean = self.observation_map(x_now)[0]
        return self.observation_distribution(y_now - mean)

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

        q is Gaussian with \Sigma^{-1} = \Sigma_state^{-1} + O'^T
        \Sigma_O^{-1} O' and \mu = G (y-O(\Phi(x_now)))

        """
        # phi is the mean of the forecast state distribution
        phi = hmmds.synthetic.filter.lorenz_sde.lorenz_integrate(
            x_now, 0.0, self.dt, self.s, self.r, self.b)
        # psi is the mean of the forecast observation distribution
        psi, d_psi = self.observation_map(phi)
        # covariance of the importance distribution
        covariance = numpy.linalg.inv(
            self.inverse_state_covariance + numpy.linalg.multi_dot(
                [d_psi.T, self.inverse_observation_covariance, d_psi]))
        gain = numpy.linalg.multi_dot(
            [covariance, d_psi.T, self.inverse_observation_covariance])
        # mean of the importance distribution
        mean = phi + numpy.dot(gain, y_next - psi)
        importance_distribution = hmm.state_space.MultivariateNormal(
            mean, covariance, self.rng)
        x_next = importance_distribution.draw()
        q_value = importance_distribution(x_next)
        return x_next, q_value

    def prior(self: LorenzSystem, x_0):
        return self.initial_distribution(x_0)


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
                        default='test_particle',
                        help='Path to store data')
    parser.add_argument('--random_seed', type=int, default=9)
    return parser.parse_args(argv)


def make_lorenz_system(args, rng):
    """Make a LorenzSystem instance

    Args:
        args: Command line arguments
        rng:

    Returns:
        (A LorenzSystem instance, an initial state, an inital distribution)

    """

    x_dim = 3
    y_dim = 1
    state_covariance = numpy.eye(x_dim) * args.state_noise**2
    observation_covariance = numpy.eye(y_dim) * args.observation_noise**2
    initial_mean = numpy.array(
        [args.x_initial_0, args.x_initial_1, args.x_initial_2])
    initial_covariance = state_covariance
    initial_distribution = hmm.state_space.MultivariateNormal(
        initial_mean, initial_covariance)
    result = LorenzSystem(args.laser_dt, args.s, args.r, args.b,
                          state_covariance, args.x_ratio, args.offset,
                          observation_covariance, initial_mean,
                          initial_covariance, rng)
    return result


def main(argv=None):
    """ Takes almost 18 minutes
    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    system = make_lorenz_system(args, rng)
    laser_data = plotscripts.introduction.laser.read_data(args.LaserData)
    n_times = 2876
    assert laser_data.shape == (2, n_times)
    observations = laser_data[1, :].astype(int).reshape((n_times, 1))

    n_particles = numpy.ones(n_times, dtype=int) * 300
    n_particles[0:3] *= 10
    particles, forward_means, forward_covariances, log_likelihood = system.forward_filter(
        observations, n_particles, threshold=0.5)
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
