"""utilities.py:

"""
from __future__ import annotations

import typing

import numpy

import hmm.state_space
import hmmds.synthetic.filter.lorenz_sde
from hmmds.synthetic.filter.lorenz_sde import lorenz_integrate


def read_tang(data_file):
    """Read one of Tang's laser data files as an array.
    """
    with open(data_file, 'r', encoding='utf-8') as file:
        lines = file.readlines(
        )  # There are only 28278 lines and memory is big and cheap

    assert lines[0].split()[0] == 'BEGIN'
    assert lines[-1].split()[-1] == 'END'
    return numpy.array([[float(x) for x in line.split()] for line in lines[1:-1]
                       ]).T


class Parameters:
    """Parameters for laser data.

    A tuple of parameters is passed to objective_funciton which uses
    this class to associate a name with each value and then invokes
    make_non_stationary with a Parameters instance as an argument.

    Subclasses could let you optimize over smaller sets of values.
    """

    # The order of names in variables must match the order in __init__
    variables = """
s r b
x_initial_0 x_initial_1 x_initial_2
t_ratio x_ratio offset
state_noise observation_noise""".split()

    # Use natural names.  Values are assigned in loop over
    # self.variables.

    # pylint: disable = invalid-name, unused-argument, too-many-arguments
    def __init__(
        self,
        s,
        r,
        b,
        x_initial_0,
        x_initial_1,
        x_initial_2,
        t_ratio,
        x_ratio,
        offset,
        state_noise=0.7,
        observation_noise=0.5,
        # The following are not subject to optimization
        fudge=1.0,
        laser_dt=0.04,
    ):
        var_dict = vars()
        for name in self.variables:
            setattr(self, name, var_dict[name])
        self.fudge = fudge  # Roll into state_noise
        self.laser_dt = laser_dt

    def set_initial_state(self, initial_state):
        self.x_initial_0, self.x_initial_1, self.x_initial_2 = initial_state

    def values(self: Parameters):
        """Make a tuple out of self for use in optimization code.
        """
        return tuple(getattr(self, key) for key in self.variables)

    def __str__(self: Parameters):
        """Make a string of key value pairs.
        """
        result = ''
        for key in self.variables:
            result += f'{key} {getattr(self,key)}\n'
        return result

    def write(self: Parameters, path):
        """Write self to a file given by path.
        """
        with open(path, 'w', encoding='utf-8') as file_:
            file_.write(self.__str__())


def read_parameters(path: str) -> Parameters:
    """Read values from text file and return Parameters instance
    """
    in_dict = {}
    with open(path, 'r', encoding='utf-8') as file_:
        for line in file_.readlines():
            parts = line.split()
            if parts[0] in Parameters.variables:  # Skip result strings
                in_dict[parts[0]] = float(parts[1])
    value_list = [in_dict[name] for name in Parameters.variables]
    return Parameters(*value_list)


def simulate(parameters: Parameters, n_samples: int) -> numpy.ndarray:
    """Integrate the Lorenz system for n_samples without noise.

    Args:
        parameters: System parameters
        n_samples: Number of samples to return

    Return:
        A numpy array n_samples x 3
"""
    t_sample = 0.04 * parameters.t_ratio
    samples = numpy.empty((n_samples, 3))
    samples[0] = numpy.array([
        parameters.x_initial_0, parameters.x_initial_1, parameters.x_initial_2
    ])
    for i in range(1, n_samples):
        # No typing for lorenz_sde.  pylint: disable = c-extension-no-member
        samples[i] = hmmds.synthetic.filter.lorenz_sde.lorenz_integrate(
            samples[i - 1],
            0,
            t_sample,
            parameters.s,
            parameters.r,
            parameters.b,
            h_max=1.0e-3)
    return samples


def observe(parameters, n_samples):
    """Simulate noiseless observations of noiseless dynamics.
    """
    states = simulate(parameters, n_samples)
    return (parameters.x_ratio * states[:, 0]**2 +
            parameters.offset).astype(int)


class FixedPoint:
    """Characterizes a focus of the Lorenz system

    Args:
        r,s,b: Parameters of Lorenz system
        sign: Specifies which focus to characterize
    """

    def __init__(
            self,  # FixedPoint
            r=28.0,
            s=10.0,
            b=8.0 / 3,
            sign=1,
    ):
        assert abs(sign) == 1
        self.r = r
        root = sign*numpy.sqrt(b * (r - 1))
        self.fixed_point = numpy.array([root, root, r - 1])
        df_dx = numpy.array([  # derivative of x_dot wrt x
            [-s, s, 0], [1, -1, -root], [root, root, -b]
        ])
        values, right_vectors = numpy.linalg.eig(df_dx)
        left_vectors = numpy.linalg.inv(right_vectors)
        for i in range(3):
            assert numpy.allclose(numpy.dot(left_vectors[i], df_dx),
                                  values[i] * left_vectors[i])
            assert numpy.allclose(numpy.dot(df_dx, right_vectors[:, i]),
                                  values[i] * right_vectors[:, i])
        assert values[
            0].imag == 0.0, f"First eigenvalue is not real: values={values}"
        self.projection = numpy.dot(right_vectors[:, 1:],
                                    left_vectors[1:, :]).real
        # projection onto subspace of complex eigenvectors
        self.image_2d = numpy.dot(numpy.array([[1, 0, 0], [0, 0, 1]]),
                                  self.projection)
        # Components 0 and 2 of projection
        assert numpy.allclose(numpy.dot(self.projection, right_vectors[:, -1]),
                              right_vectors[:, -1])
        self.omega = numpy.abs(values[-1].imag)
        self.period = 2 * numpy.pi / self.omega
        self.relax = values[-1].real

    def initial_state(
            self,  # FixedPoint
            delta_x):
        """Find initial state that is distance delta_x from fixed point
        """
        coefficients = numpy.linalg.lstsq(self.image_2d,
                                          numpy.array([delta_x, 0]),
                                          rcond=None)[0]
        return numpy.dot(self.projection, coefficients) + self.fixed_point

    def map_time(
            self,  # FixedPoint
            x_initial):
        """Find time and position that x_initial maps to x[2] = r-1
        """
        h_max = 1.0e-3
        tenths = numpy.empty((20, 3))
        t_step = self.period / 10
        # Integrate at least once because x_initial is on boundary
        x_last = lorenz_integrate(x_initial, 0, t_step, h_max=h_max, r=self.r)
        for i in range(20):
            x_next = lorenz_integrate(x_last, 0, t_step, h_max=h_max, r=self.r)
            if x_next[2] > self.r - 1 > x_last[2]:
                break
            x_last = x_next
        else:
            raise RuntimeError("Failed to find bracket")

        def func(time):
            x_time = lorenz_integrate(x_last, 0, time, h_max=h_max, r=self.r)
            result = x_time[2] - (self.r - 1)
            return result

        delta_t = scipy.optimize.brentq(func, 0, t_step)
        t_final = (i + 1) * t_step + delta_t
        x_final = lorenz_integrate(x_last, 0, delta_t, h_max=h_max, r=self.r)
        return t_final, x_final


def make_non_stationary(parameters, rng):
    """Make an SDE system instance

    Args:
        parameters: A Parameters instance
        rng: Dummy

    Returns:
        (An SDE instance, an initial state, an inital distribution)

    """

    # The next three functions are passed to SDE.__init__
    # Will use t, s, r, b pylint: disable = invalid-name
    def dx_dt(_, x, s, r, b):
        return numpy.array([
            s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]
        ])

    def tangent(t, x_dx, s, r, b):
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
        y = numpy.array([parameters.x_ratio * state[0]**2 + parameters.offset])
        dy_dx = numpy.array([[parameters.x_ratio * 2 * state[0], 0, 0]])
        return y, dy_dx

    x_dim = 3
    state_noise = numpy.ones(x_dim) * parameters.state_noise
    y_dim = 1
    observation_noise = numpy.eye(y_dim) * parameters.observation_noise

    # lorenz_sde.SDE only uses Cython for methods forecast and simulate
    dt = parameters.laser_dt * parameters.t_ratio  # pylint: disable = invalid-name
    h_max = 1.0e-3
    # No type info for lorenz_sde pylint: disable = c-extension-no-member
    system = hmmds.synthetic.filter.lorenz_sde.SDE(
        dx_dt,
        tangent,
        state_noise,
        observation_function,
        observation_noise,
        dt,
        x_dim,
        ivp_args=(parameters.s, parameters.r, parameters.b, h_max),
        fudge=parameters.fudge)
    initial_mean = numpy.array([
        parameters.x_initial_0, parameters.x_initial_1, parameters.x_initial_2
    ])
    initial_covariance = numpy.outer(state_noise, state_noise)
    initial_distribution = hmm.state_space.MultivariateNormal(
        initial_mean, initial_covariance)
    result = hmm.state_space.NonStationary(system, dt, rng)
    return result, initial_distribution, initial_mean
