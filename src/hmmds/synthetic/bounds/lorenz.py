"""lorenz.py

"""
from __future__ import annotations  # Enables, eg, (self: LocalNonStationary

import sys
import typing

import numpy
import numpy.random
import numpy.linalg
import scipy.special

import hmm.state_space
import hmmds.synthetic.filter.lorenz_sde


def positive_definite(x):
    """Assert that x is positive definite
    """
    symmetric = (x + x.T) / 2
    assert numpy.allclose(x, symmetric)
    vals, vecs = numpy.linalg.eigh(symmetric)
    assert vals.min() >= 0.0, f'vals={vals}'


class LocalNonStationary(hmm.state_space.NonStationary):
    """Overwrite simulate_n_steps method for quantized observations,
    and overwrite forward_filter method to return more than updated
    means and covariances.

    """

    def __init__(self: LocalNonStationary, system: hmm.state_space.SDE,
                 dt: float, y_step: float, rng: numpy.random.Generator):
        super().__init__(system, dt, rng)
        self.y_step = y_step

    def simulate_n_steps(
            self: LocalNonStationary,
            initial_dist: hmm.state_space.MultivariateNormal,
            n_samples: int,
            states_0=None) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """Return simulated sequences of states and observations.


        Args:
            initial_dist: Distribution of initial state
            n_samples: Number of states and observations to return
            y_setp: Distance between allowed y_values

        Differs from parent by Quantizing the observations
        """
        xs, ys = self.system.simulate_n_steps(initial_dist, n_samples, states_0)
        return xs, numpy.floor(ys / self.y_step) * self.y_step + self.y_step / 2

    def forward_filter(self: LocalNonStationary,
                       initial_dist: hmm.state_space.MultivariateNormal,
                       y_array: numpy.ndarray):
        """Run Kalman filter on observations y_array.

        Args:
            initial_dist: Prior for state
            y_array: Sequence of observations

        Differs from parent hmm.state_space.LinearStationary by
        returning more than update distributions.


        """
        forecast_means = numpy.empty((len(y_array), self.x_dim))
        forecast_covariances = numpy.empty(
            (len(y_array), self.x_dim, self.x_dim))
        update_means = numpy.empty((len(y_array), self.x_dim))
        update_covariances = numpy.empty((len(y_array), self.x_dim, self.x_dim))
        y_means = numpy.empty(len(y_array))
        y_variances = numpy.empty(len(y_array))
        y_probabilities = numpy.empty(len(y_array))

        forecast_distribution = initial_dist
        for t, y in enumerate(y_array):
            forecast_means[t] = forecast_distribution.mean
            forecast_covariances[t] = forecast_distribution.covariance

            forecast_distribution, update_distribution, y_forecast = self.update_step(
                forecast_distribution, y)
            # update_distribution and y_forecast are for t.
            # forecast_distribution is for t+1

            update_means[t] = update_distribution.mean
            update_covariances[t] = update_distribution.covariance

            y_means[t] = y_forecast.mean[0]
            y_variances[t] = y_forecast.covariance[0, 0]

            def cumulative_prob(y, mean=y_means[t], variance=y_variances[t]):
                """return 1/2 [1+erf(z/sqrt(2))]
                """
                return (1 + scipy.special.erf(
                    (y - mean) / numpy.sqrt(2 * variance))) / 2

            # y is on a quantization level, and y_means[t] is not.
            # The forecast distribution is Normal(y_means[t],
            # y_variances[t]).  The likelihood of y is the probability
            # of the interval y +/- y_step/2
            y_probabilities[t] = cumulative_prob(
                y + self.y_step / 2) - cumulative_prob(y - self.y_step / 2)

        return forecast_means, forecast_covariances, update_means, update_covariances, y_means, y_variances, y_probabilities

    def update_step(self: LocalNonStationary, prior, y):
        """Calculate new x_dist based on observation y then forecast

        Args:
            prior: Prior for state based on past ys
            y: Current observation

        Returns:
            (Forecast for y_now,
            a posteriori distribution for state_now,
            and forecast for state_next)

        Differs from parent forward_step in goals and return
        values. FixMe: Figure if parent is wrong.

        """
        t = 0.0  # The lorenz system is stationary

        g_t, d_g, observation_noise_covariance = self.system.update(
            prior.mean, t)

        # y_forecast is the distribution of y[t]|y[:t]
        cov = observation_noise_covariance + numpy.dot(
            numpy.dot(d_g, prior.covariance), d_g.T)
        y_forecast = hmm.state_space.MultivariateNormal(g_t, cov, self.rng)

        # Calculate distribution of x[t]|y[:t+1], ie, include y in history
        update = self.update(prior,
                             y,
                             d_g,
                             observation_noise_covariance,
                             observation_mean=g_t)

        # Integrate updated mean forward
        f_next, d_f, state_noise_covariance = self.system.forecast(
            update.mean, t, self.dt)

        # forecast ~ Normal(f_next, state_noise_covaraince + d_f Sigma_update d_f.T)
        forecast = self.forecast(update,
                                 d_f,
                                 state_noise_covariance,
                                 new_mean=f_next)
        positive_definite(forecast.covariance)

        return forecast, update, y_forecast


def make_system(s: float, r: float, b: float, unit_state_noise_scale: float,
                observation_noise_scale: float, dt: float, y_step: float,
                fudge: float, h_max: float, atol: float,
                rng: numpy.random.Generator):
    """Make two LocalNonStationary instances based on a Lorenz SDE

    Args:
        s, r, b: Parameters of Lorenz ODE
        unit_state_noise_scale: sqrt(dt)*this*std_normal(x_dim) = noise
        observation_noise_scale: this*std_normal(y_dim) = noise
        dt: Sample interval
        fudge: Ratio of system noise covariance filtering/generation
        y_step: Quantization step size
        h_max: Maximum time step for cython integrator
        atol: Error bound for scipy integrator
        rng: Random number generator

    Returns: dict

    'Cython': based on hmmds.synthetic.filter.lorenz_sde.SDE instance,
    'SciPy': based on hmm.state_space.SDE instance
    'stationary_distribution': hmm.state_space.MultivariateNormal instance
    'initial_state': numpy.ndarray

    Derived from hmmds.synthetic.filter.lorenz_simulation

    """

    # The next three functions are passed to SDE.__init__
    # pylint: disable = invalid-name

    def dx_dt(_, x, s, r, b):
        """Calculate the Lorenz vector field at x
        """
        return numpy.array([
            s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]
        ])

    def tangent(t, x_dx, s, r, b):
        """Calculate the Lorenz vector field and its tangent at x
        """
        result = numpy.empty(12)  # Allocate storage for result

        # Unpack state and derivative from argument
        x = x_dx[:3]
        dx_dx0 = x_dx[3:].reshape((3, 3))

        # First three components are the value of the vector field F(x)
        result[:3] = dx_dt(t, x, s, r, b)

        dF = numpy.array([  # The derivative of F wrt x
            [-s, s, 0], [r - x[2], -1, -x[0]], [x[1], x[0], -b]
        ])

        # Assign the tangent part of the return value.
        result[3:] = numpy.dot(dF, dx_dx0).reshape(-1)

        return result

    def observation_function(
            _, state) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """Calculate observation and its derivative
        """
        observation_map = numpy.array([[1, 0, 0]])
        return numpy.dot(observation_map, state), observation_map

    x_dim = 3
    unit_state_noise_map = numpy.eye(x_dim) * unit_state_noise_scale
    y_dim = observation_function(0, numpy.ones(x_dim))[0].shape[0]
    observation_noise_map = numpy.eye(y_dim) * observation_noise_scale

    # pylint: disable = c-extension-no-member, duplicate-code
    cython = hmmds.synthetic.filter.lorenz_sde.SDE(dx_dt,
                                                   tangent,
                                                   unit_state_noise_map,
                                                   observation_function,
                                                   observation_noise_map,
                                                   dt,
                                                   x_dim,
                                                   ivp_args=(s, r, b, h_max),
                                                   rng=rng,
                                                   fudge=fudge)
    sde = hmm.state_space.SDE(dx_dt,
                              tangent,
                              unit_state_noise_map,
                              observation_function,
                              observation_noise_map,
                              dt,
                              x_dim,
                              ivp_args=(s, r, b),
                              rng=rng,
                              atol=atol,
                              fudge=fudge)
    result = {}  # Collection of items to return
    relaxed = sde.relax(500)[0]  # Relax to attractor
    result['initial_state'], result['stationary_distribution'] = sde.relax(
        500, initial_state=relaxed)  # Collect data for distribution

    result['Cython'] = LocalNonStationary(cython, dt, y_step, rng)
    result['SciPy'] = LocalNonStationary(sde, dt, y_step, rng)
    return result


def main(argv=None):
    s = 10.0
    r = 28.0
    b = 8.0 / 3

    rng = numpy.random.default_rng(3)

    dev_state_noise = 1.0e-6
    dev_observation_noise = .01
    d_t = 0.25
    n_times = 500
    y_step = 1.0e-4
    h_max = 1.0e-3
    atol = 1.0e-7
    fudge = 1.0

    ivp_args = (s, r, b)

    # Make two LocalNonStationary instances and initialization data

    made_dict = make_system(s, r, b, dev_state_noise, dev_observation_noise,
                            d_t, y_step, fudge, h_max, atol, rng)
    # Unpack made_dict
    cython_ = made_dict['Cython']
    scipy_ = made_dict['SciPy']
    initial_state = made_dict['initial_state']
    stationary_covariance = made_dict['stationary_distribution'].covariance

    # Asssert f and tangent are the same for Cython and SciPy
    x_dx = numpy.linspace(.1, 1.2, 12)
    assert numpy.allclose(cython_.system.f(0, x_dx[:3], s, r, b),
                          scipy_.system.f(0, x_dx[:3], s, r, b))
    assert numpy.allclose(cython_.system.tangent(0, x_dx, s, r, b),
                          scipy_.system.tangent(0, x_dx, s, r, b))

    initial_distribution = hmm.state_space.MultivariateNormal(
        initial_state, stationary_covariance, rng)
    x, y = scipy_.simulate_n_steps(initial_distribution, n_times, y_step)
    # Check that forward_filter runs
    forecast_means, forecast_covariances, update_means, update_covariances, y_means, y_variances, y_probabilities = scipy_.forward_filter(
        initial_distribution, y)
    return 0


if __name__ == "__main__":
    sys.exit(main())
