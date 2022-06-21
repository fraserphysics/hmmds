"""optimize_ekf.py: Find parameters of extended Kalman filter for
laser data points.

"""
from __future__ import annotations

import sys
import typing
import argparse
import pickle

import numpy
import scipy.optimize

import hmm.state_space
import hmmds.synthetic.filter.lorenz_sde

import explore
import plotscripts.introduction.laser


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Optimize parameters for laser data')
    parser.add_argument('--parameter_type',
                        type=str,
                        default='parameter',
                        help='parameter or GUI_out')
    parser.add_argument('--laser_data',
                        type=str,
                        default='LP5.DAT',
                        help='path of data file')
    parser.add_argument('--length',
                        type=int,
                        default=2876,
                        help='optimize over this number of data samples')
    parser.add_argument('--method',
                        type=str,
                        default='Powell',
                        help='Argument to scipy.optimize.minimize or "skip"')
    parser.add_argument('--plot_data', type=str, help='Path to store data')
    parser.add_argument('parameters_in_out',
                        type=str,
                        help='paths to files',
                        nargs='+')
    return parser.parse_args(argv)


def explore_to_parameters(in_path="explore.txt"):
    """Use data that GUI explore.py wrote to create a Parameters instance."""
    in_dict = {}
    with open(in_path, 'r') as file_:
        for line in file_.readlines():
            name, value_str = line.split()
            in_dict[name] = float(value_str)
    s = 10.0
    r = in_dict['r']
    b = 8.0 / 3
    fixed_point = explore.FixedPoint(r)
    initial_state = hmmds.synthetic.filter.lorenz_sde.lorenz_integrate(
        fixed_point.initial_state(in_dict['delta_x']), 0.0, in_dict['delta_t'],
        s, r, b)
    parameters = Parameters(
        s,
        r,
        b,
        *initial_state,
        in_dict['t_ratio'],
        in_dict['x_ratio'],
        in_dict['offset'],
    )
    return parameters


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

    def values(self: Parameters):
        return tuple(getattr(self, key) for key in self.variables)

    def __str__(self: Parameters):
        result = ''
        for key in self.variables:
            result += f'{key} {getattr(self,key)}\n'
        return result

    def write(self: Parameters, path):
        with open(path, 'w') as file_:
            file_.write(self.__str__())


def read_parameters(path):
    in_dict = {}
    with open(path, 'r') as file_:
        for line in file_.readlines():
            parts = line.split()
            if parts[0] in Parameters.variables:  # Skip result strings
                in_dict[parts[0]] = float(parts[1])
    value_list = [in_dict[name] for name in Parameters.variables]
    return Parameters(*value_list)


def objective_function(values_in, laser_data, parameter_class):
    """For optimization"""
    parameter = parameter_class(*values_in)
    non_stationary, initial_distribution, initial_state = make_non_stationary(
        parameter, None)
    result = non_stationary.log_likelihood(initial_distribution, laser_data)
    print(f"""objective_function = {result}""")

    return -result


def make_non_stationary(parameters, rng):
    """Make an SDE system instance

    Args:
        parameters: A Parameters instance
        rng: Dummy

    Returns:
        (An SDE instance, an initial state, an inital distribution)

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
        y = numpy.array([parameters.x_ratio * state[0]**2 + parameters.offset])
        dy_dx = numpy.array([[parameters.x_ratio * 2 * state[0], 0, 0]])
        return y, dy_dx

    x_dim = 3
    state_noise = numpy.ones(x_dim) * parameters.state_noise
    y_dim = 1
    observation_noise = numpy.eye(y_dim) * parameters.observation_noise

    # lorenz_sde.SDE only uses Cython for methods forecast and simulate
    dt = parameters.laser_dt * parameters.t_ratio
    system = hmmds.synthetic.filter.lorenz_sde.SDE(dx_dt,
                                                   tangent,
                                                   state_noise,
                                                   observation_function,
                                                   observation_noise,
                                                   dt,
                                                   x_dim,
                                                   ivp_args=(parameters.s,
                                                             parameters.r,
                                                             parameters.b),
                                                   fudge=parameters.fudge)
    initial_mean = numpy.array([
        parameters.x_initial_0, parameters.x_initial_1, parameters.x_initial_2
    ])
    initial_covariance = numpy.outer(state_noise, state_noise)
    initial_distribution = hmm.state_space.MultivariateNormal(
        initial_mean, initial_covariance)
    result = hmm.state_space.NonStationary(system, dt, rng)
    return result, initial_distribution, initial_mean


# Powell, BFGS, Nelder-Mead
def optimize(initial_parameters, laser_data, method='Powell', options={}):

    defaults = initial_parameters.values()
    result = scipy.optimize.minimize(
        objective_function,
        defaults,
        method=method,
        options=options,
        args=(laser_data, Parameters),
    )
    parameters_max = Parameters(*result.x)
    print(f"""
parameters_max:
{parameters_max}
f_max={-result.fun}
success={result.success}
message={result.message}
iterations={result.nit}""")
    return parameters_max, result


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]
    args = parse_args(argv)

    if not args.plot_data and len(args.parameters_in_out) != 2:
        raise RuntimeError('No file specified for results')
    if len(args.parameters_in_out) > 2:
        raise RuntimeError('More than 2 positional arguments')

    if args.parameter_type == 'GUI_out':
        parameters = explore_to_parameters(args.parameters_in_out[0])
    elif args.parameter_type == 'parameter':
        parameters = read_parameters(args.parameters_in_out[0])
    else:
        raise RuntimeError(
            f'parameter_type {args.parameter_type} not recognized')

    laser_data_y_t = plotscripts.introduction.laser.read_data(args.laser_data)
    assert laser_data_y_t.shape == (2, 2876)
    # Put y values in global
    laser_data = laser_data_y_t[1, :].astype(int).reshape((2876, 1))

    if args.method != 'skip':
        #options = {'maxiter': 2}
        parameters_max, result = optimize(
            parameters,
            laser_data[:args.length],
            method=args.method,
            #options=options
        )
    else:
        parameters_max = parameters
    if len(args.parameters_in_out) == 2:
        parameters_max.write(args.parameters_in_out[1])
    if args.method != 'skip' and len(args.parameters_in_out) == 2:
        with open(args.parameters_in_out[1], 'a') as _file:
            _file.write(f"""f_max {-result.fun}
success {result.success}
iterations {result.nit}
message {result.message}
n_data {args.length}""")

    if args.plot_data is None:
        return 0
    sde, initial_distribution, initial_state = make_non_stationary(
        parameters_max, None)
    forward_means, forward_covariances = sde.forward_filter(
        initial_distribution, laser_data)
    cross_entropy = sde.log_likelihood(initial_distribution,
                                       laser_data) / len(laser_data)
    print(f'cross_entropy {cross_entropy}')

    with open(args.plot_data, 'wb') as _file:
        pickle.dump(
            {
                'dt': parameters_max.laser_dt,
                'observations': laser_data,
                'forward_means': forward_means,
                'forward_covariances': forward_covariances,
                'cross_entropy': cross_entropy
            }, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
