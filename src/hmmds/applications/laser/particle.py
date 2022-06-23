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

from hmmds.applications.laser import optimize_ekf
import hmmds.applications.laser.utilities


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
        Args:
           y_now: The current observation
           x_now: A particle current state
        """
        delta_y = y_now - self.observation_map(x_now)[0]
        return self.observation_distribution(delta_y), delta_y

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

    def forward_filter(self: LorenzSystem,
                       y_array: numpy.ndarray,
                       n_particles: int | numpy.ndarray,
                       prior: typing.Callable | None = None,
                       threshold: float = 1.0):
        """Run filter on observations y_array

        Args:
            y_array:
            n_particles: Single int or array
            prior:
            threshold: Resample if effective_sample_size < threshold * n_particles

        Returns:
            (particles, means, covariances, log_likelihood)

        log_likelihood = log(prob(y[0:n_times]|model))
        Note: particles.shape=(N_particles, N_observations)
        """
        if prior is None:
            prior = self.prior
        n_times, check_dim = y_array.shape
        assert check_dim == self.y_dimension

        if isinstance(n_particles, int):
            n_particles = numpy.ones(n_times, dtype=int) * n_particles
        assert n_particles.dtype == numpy.dtype('int64')
        assert n_particles.shape == (n_times,)

        delta_ys = numpy.zeros((n_times, self.y_dimension))

        weights = numpy.empty(n_particles[0])

        means = numpy.empty((n_times, self.x_dimension))
        covariances = numpy.empty((n_times, self.x_dimension, self.x_dimension))

        # Initialize at t=0
        particles = numpy.empty((n_particles[0], n_times, self.x_dimension))

        # Draw particles and calculate weights for EV_{prior}
        for i in range(n_particles[0]):
            # x_i_0 is a draw from q, and q_i_0 is q(x_i_0)
            x_i_0, q_i_0 = self.importance_0(y_array[0])
            weights[i] = prior(x_i_0) / q_i_0
            particles[i, 0, :] = x_i_0
        weights /= weights.sum()
        # Now EV_{prior} f(x_0) \approx \sum_i weights[i] f(particles[i,0,:])

        likelihood_0 = 0
        # likelihood_0 = EV_{prior} p(y_0|x_0)

        # Calculate likelihood_0 and weights for EV_{x_0|y_0}
        for i, particle in enumerate(particles):
            x_i_0 = particle[0, :]
            likelihood_i, _ = self.observation(y_array[0], x_i_0)
            likelihood_0 += weights[i] * likelihood_i
            weights[i] *= likelihood_i
        weights = weights / weights.sum()

        # Finish work for t=0
        log_like = numpy.log(likelihood_0)
        means[0, :], covariances[0, :, :] = hmm.particle.moments(
            particles[:, 0, :], weights)
        particles, weights = hmm.particle.resample(particles, weights, self.rng,
                                                   n_particles[0])

        # Iterate t_previous=0, ..., t_previous=T-2
        for t_previous, y_t in enumerate(y_array[1:]):
            t_now = t_previous + 1
            likelihood_now = 0

            normalization_likelihood = 0
            # Draw particles and calculate weights for EV_{x[t_now]|y[:t_now]}
            for i, particle in enumerate(particles):
                predecessor = particle[t_previous]
                x_i_t, q_i_t = self.importance(y_t, predecessor)
                particles[i, t_now, :] = x_i_t
                weights[i] *= self.transition(x_i_t, predecessor) / q_i_t

                # Calculate likelihood_now and weights for EV_{x[t_now]|y[:t_now+1]}
                x_i_t = particle[t_now, :]
                likelihood_i, delta_y_i = self.observation(
                    y_array[t_now], x_i_t)
                delta_ys[t_now] += delta_y_i * weights[i]
                normalization_likelihood += weights[i]
                likelihood_now += weights[i] * likelihood_i
                weights[i] *= likelihood_i

            # Finish up work for t=t_now
            delta_ys[t_now] /= normalization_likelihood
            log_like += numpy.log(likelihood_now / normalization_likelihood)
            weights = weights / weights.sum()
            means[t_now, :], covariances[t_now, :, :] = hmm.particle.moments(
                particles[:, t_now, :], weights)
            particles, weights = hmm.particle.resample(particles, weights,
                                                       self.rng,
                                                       n_particles[t_now],
                                                       threshold)
        return particles, means, covariances, log_like, delta_ys


def parse_args(argv):
    """Define parser and parse command line.

    """

    parser = argparse.ArgumentParser(description='Filter laser data')
    parser.add_argument('--laser_data',
                        type=str,
                        default='LP5.DAT',
                        help='Path to laser data')
    parser.add_argument('--random_seed', type=int, default=9)
    parser.add_argument('--n_times', type=int, default=2876)
    parser.add_argument('parameters_in', type=str, help='path to file')
    parser.add_argument('result', type=str, help='Path to store data')
    return parser.parse_args(argv)


def make_lorenz_system(parameters, rng):
    """Make a LorenzSystem instance

    Args:
        parameters: Values read from file
        rng:

    Returns:
        (A LorenzSystem instance, an initial state, an inital distribution)

    """

    x_dim = 3
    y_dim = 1
    dt = parameters.laser_dt * parameters.t_ratio
    state_covariance = numpy.eye(x_dim) * parameters.state_noise**2
    observation_covariance = numpy.eye(y_dim) * parameters.observation_noise**2
    initial_mean = numpy.array([
        parameters.x_initial_0, parameters.x_initial_1, parameters.x_initial_2
    ])
    initial_covariance = state_covariance
    initial_distribution = hmm.state_space.MultivariateNormal(
        initial_mean, initial_covariance)
    result = LorenzSystem(dt, parameters.s, parameters.r, parameters.b,
                          state_covariance, parameters.x_ratio,
                          parameters.offset, observation_covariance,
                          initial_mean, initial_covariance, rng)
    return result


def main(argv=None):
    """ Takes almost 18 minutes
    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    parameters = hmmds.applications.laser.utilities.read_parameters(
        args.parameters_in)
    system = make_lorenz_system(parameters, rng)
    laser_data = hmmds.applications.laser.utilities.read_tang(args.laser_data)
    assert laser_data.shape == (2, 2876)
    observations = laser_data[1, :args.n_times].astype(int).reshape(
        (args.n_times, 1))

    n_particles = numpy.ones(args.n_times, dtype=int) * 300
    n_particles[0:3] *= 10
    particles, forward_means, forward_covariances, log_likelihood, delta_ys = system.forward_filter(
        observations, n_particles, threshold=0.5)
    print(f'log_likelihood={log_likelihood}')

    with open(args.result, 'wb') as _file:
        pickle.dump(
            {
                'dt': parameters.laser_dt,
                'observations': observations,
                'forward_means': forward_means,
                'forward_covariances': forward_covariances,
                'delta_ys': delta_ys,
            }, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
