"""lorenz.py

"""
import typing

import numpy
import numpy.random

import hmm.state_space
import hmmds.synthetic.filter.lorenz_sde


class LocalNonStationary(hmm.state_space.NonStationary):
    """Overwrite forward_filter method so that it returns both forecast
    and update distributions

    """

    def forward_filter(self, initial_dist, y_array):
        """attach both forcast and update distributions to self
        replaces hmm.state_space.LinearStationary.forward_filter
        """
        forecast_means = numpy.empty((len(y_array), self.x_dim))
        forecast_covariances = numpy.empty((len(y_array), self.x_dim, self.x_dim))
        update_means = numpy.empty((len(y_array), self.x_dim))
        update_covariances = numpy.empty((len(y_array), self.x_dim, self.x_dim))
        update_distribution = initial_dist
        for t, y in enumerate(y_array):
            forecast_distribution, update_distribution = self.forward_step(update_distribution, y)
            forecast_means[t] = forecast_distribution.mean
            forecast_covariances[t] = forecast_distribution.covariance
            update_means[t] = update_distribution.mean
            update_covariances[t] = update_distribution.covariance
        return forecast_means, forecast_covariances, update_means, update_covariances

    def forward_step(self, prior, y):
        """save or return forecast
        replaces hmm.state_space.NonStationary.forward_step
        """
        f_t, d_f, state_noise_covariance = self.system.forecast(
            prior.mean, 0.0, self.dt)
        g_t, d_g, observation_noise_covariance = self.system.update(f_t, 0.0)

        forecast = self.forecast(prior,
                                 d_f,
                                 state_noise_covariance,
                                 new_mean=f_t)

        update = self.update(forecast,
                             y,
                             d_g,
                             observation_noise_covariance,
                             observation_mean=g_t)
        return forecast, update


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
    state_noise_map = numpy.eye(x_dim) * unit_state_noise_scale
    y_dim = observation_function(0, numpy.ones(x_dim))[0].shape[0]
    observation_noise_map = numpy.eye(y_dim) * observation_noise_scale

    # pylint: disable = c-extension-no-member, duplicate-code
    sde = hmmds.synthetic.filter.lorenz_sde.SDE(dx_dt,
                                                tangent,
                                                state_noise_map,
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
