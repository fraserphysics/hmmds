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
    symmetric = (x + x.T) / 2
    assert numpy.allclose(x, symmetric)
    vals, vecs = numpy.linalg.eigh(symmetric)
    assert vals.min() >= 0.0, f'vals={vals}'


class LocalNonStationary(hmm.state_space.NonStationary):
    """Overwrite forward_filter method so that it returns both forecast
    and update distributions  Also return probabilities.

    Overwrite NonStationary.simulate_n_steps to quantize y

    """

    def simulate_n_steps(
            self: LocalNonStationary,
            initial_dist: hmm.state_space.MultivariateNormal,
            n_samples: int,
            y_step: float,
            states_0=None) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """Return simulated sequences of states and observations.


        Args:
            initial_dist: Distribution of initial state
            n_samples: Number of states and observations to return
            y_setp: Distance between allowed y_values

        Differs from parent by Quantizing the observations
        """
        xs, ys = self.system.simulate_n_steps(initial_dist, n_samples, states_0)
        return xs, numpy.floor(ys / y_step) * y_step + y_step / 2

    def forward_filter(self: LocalNonStationary, initial_dist, y_array, y_step):
        """Run Kalman filter on observations y_array.

        Args:
            initial_dist: Prior for state
            y_array: Sequence of observations

        Differs from parent hmm.state_space.LinearStationary by
        returning both forcast and update distributions.

        ToDo: Assume y.shape = (1,).  Accept addtional argument
        y_step.  Calculate and return log prob y[t]|y[:t]

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

            def cumulative_prob(y, mean, variance):
                """return 1/2 [1+erf(z/sqrt(2))]
                """
                return (1 + scipy.special.erf(
                    (y - mean) / numpy.sqrt(2 * variance))) / 2

            y_probabilities[t] = cumulative_prob(
                y + y_step / 2, y_means[t], y_variances[t]) - cumulative_prob(
                    y - y_step / 2, y_means[t], y_variances[t])

        return forecast_means, forecast_covariances, update_means, update_covariances, y_means, y_variances, y_probabilities

    def update_step(self: LocalNonStationary, prior, y):
        """Calculate new x_dist based on observation y then forecast

        Args:
            prior: Prior for state based on past ys
            y: Current observation

        Returns:
            (Forecast for y_now, a posteriori distribution for state_now, and forecast for state_next)

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
                observation_noise_scale: float, dt: float,
                rng: numpy.random.Generator):
    """Make a LocalNonStationary instance based on a Lorenz SDE

    Args:
        s, r, b: Parameters of Lorenz ODE
        unit_state_noise_scale: sqrt(dt)*this*std_normal(x_dim) = noise
        observation_noise_scale: this*std_normal(y_dim) = noise
        dt: Sample interval
        rng: Random number generator

    Returns: (An hmmds.synthetic.filter.lorenz_sde.SDE instance, an
        inital distribution, an initial state)

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
    sde = hmmds.synthetic.filter.lorenz_sde.SDE(dx_dt,
                                                tangent,
                                                unit_state_noise_map,
                                                observation_function,
                                                observation_noise_map,
                                                dt,
                                                x_dim,
                                                ivp_args=(s, r, b))
    initial_state = sde.relax(500)[0]  # Relax to attractor
    final_state, stationary_distribution = sde.relax(
        500, initial_state=initial_state)  # Collect data for distribution
    return LocalNonStationary(sde, dt,
                              rng), stationary_distribution, final_state


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

    system, stationary_distribution, initial_state = make_system(
        s, r, b, dev_state_noise, dev_observation_noise, d_t, rng)
    initial_distribution = hmm.state_space.MultivariateNormal(
        initial_state, stationary_distribution.covariance / 1.0e4, rng)
    x, y = system.simulate_n_steps(initial_distribution, n_times, y_step)

    system, stationary_distribution, initial_state = make_system(
        s, r, b, dev_state_noise, dev_observation_noise, d_t, rng)

    forecast_means, forecast_covariances, update_means, update_covariances, y_means, y_variances, y_probabilities = system.forward_filter(
        initial_distribution, y, y_step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
