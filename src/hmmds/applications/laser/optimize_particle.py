"""optimize_particle.py: Optimize model parameters for particle filter
applied to laser data.

"""
from __future__ import annotations

import sys
import typing
import argparse
import pickle

import numpy
import scipy.optimize

import hmm.state_space
import hmm.particle
import hmmds.synthetic.filter.lorenz_sde
import hmmds.applications.laser.optimize_ekf
import hmmds.applications.laser.particle

import hmmds.applications.laser.utilities


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Optimize parameters of particle filter for laser data')
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
    parser.add_argument('--n_particles',
                        type=int,
                        default=50,
                        help='Number of particles')
    parser.add_argument('--random_seed', type=int, default=9)
    parser.add_argument('--method',
                        type=str,
                        default='Powell',
                        help='Argument to scipy.optimize.minimize or "skip"')
    parser.add_argument('--override_parameters',
                        type=str,
                        help='path to file with some parameters')
    parser.add_argument('--plot_data', type=str, help='Path to store data')
    parser.add_argument('parameters_in_out',
                        type=str,
                        help='paths to files',
                        nargs='+')
    return parser.parse_args(argv)


class Parameters(hmmds.applications.laser.utilities.Parameters):
    """Subclass for optimizing only the noise amplitudes

    """
    variables = """ state_noise observation_noise """.split()

    def __init__(self: Parameters, state_noise: float, observation_noise: float,
                 constants: hmmds.applications.laser.utilities.Parameters):
        """
        Args:
            state_noise: scalar float.   Covariance = eye(x_dim)* state_noise^2
            observation_noise: scalar float.   Covariance = eye(y_dim)* state_noise^2
            constants: Parameters that optimization will hold constant
            fudge: State noise multiplier for EKF
            laser_dt: Sample interval
"""
        # Set values from constants
        for name in constants.variables + 'fudge laser_dt'.split():
            setattr(self, name, getattr(constants, name))
        # Overwrite values from argument list
        var_dict = vars()
        for name in self.variables:
            setattr(self, name, var_dict[name])


def read_override(path, constants):
    in_dict = {}
    with open(path, 'r') as file_:
        for line in file_.readlines():
            parts = line.split()
            if parts[0] in Parameters.variables:  # Skip result strings
                in_dict[parts[0]] = float(parts[1])
    value_list = [in_dict[name] for name in Parameters.variables]
    return Parameters(*value_list, constants)


def objective_function(values_in, laser_data, n_particles, rng,
                       constants: hmmds.applications.laser.utilities.Parameters,
                       parameter_class):
    """For optimization

    Args:
        values_in: Passed by scipy.optimize.minimize
        laser_data: First element of args from minimize
        n_particles: Second element of args
        rng: Random number generator. Third element of args
        constants: Parameters not optimized over.  Last element of args
"""
    parameter = parameter_class(*values_in, constants)
    lorenz_system = hmmds.applications.laser.particle.make_lorenz_system(
        parameter, rng)
    numpy.seterr(divide='raise')
    try:
        particles, forward_means, forward_covariances, log_likelihood, delta_ys = lorenz_system.forward_filter(
            laser_data, n_particles, threshold=0.5)
    except FloatingPointError:
        log_likelihood = -1e6
        print('caught ZeroDivisionError')
    print(f"""objective_function = {log_likelihood}""")

    return -log_likelihood


# Powell, BFGS, Nelder-Mead
def optimize(constants: hmmds.applications.laser.utilities.Parameters,
             laser_data: numpy.ndarray,
             n_particles: numpy.ndarray,
             rng: numpy.random.Generator,
             method='Powell',
             options={}):
    """
    Args:
        constants: Instance that holds all values
        laser_data: The 1-d array of measurements
        n_particles:
        rng: Random number generator
        method: Powell, Nelder-Mead, or BFGS, etc.
        options: Could have value for maxiter, etc.

"""

    parameters = Parameters(constants.state_noise, constants.observation_noise,
                            constants)
    defaults = parameters.values()
    args = (laser_data, n_particles, rng, constants, Parameters)
    result = scipy.optimize.minimize(
        objective_function,
        defaults,
        method=method,
        options=options,
        args=args,
    )
    parameters_max = Parameters(*result.x, constants)
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
        parameters = hmmds.applications.laser.optimize_ekf.explore_to_parameters(
            args.parameters_in_out[0])
    elif args.parameter_type == 'parameter':
        parameters = hmmds.applications.laser.utilities.read_parameters(
            args.parameters_in_out[0])
    else:
        raise RuntimeError(
            f'parameter_type {args.parameter_type} not recognized')
    if args.override_parameters:
        parameters = read_override(args.override_parameters, parameters)

    laser_data_y_t = hmmds.applications.laser.utilities.read_tang(
        args.laser_data)
    assert laser_data_y_t.shape == (2, 2876)
    # Put y values in global
    laser_data = laser_data_y_t[1, :].astype(int).reshape((2876, 1))
    rng = numpy.random.default_rng(args.random_seed)

    if args.method != 'skip':
        #options = {'maxiter': 2}
        parameters_max, result = optimize(
            parameters,
            laser_data[:args.length],
            args.n_particles,
            rng,
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
    lorenz_system = hmmds.applications.laser.particle.make_lorenz_system(
        parameters_max, rng)
    particles, forward_means, forward_covariances, log_likelihood, delta_ys = lorenz_system.forward_filter(
        laser_data, args.n_particles, threshold=0.5)
    cross_entropy = log_likelihood / len(laser_data)

    with open(args.plot_data, 'wb') as _file:
        pickle.dump(
            {
                'dt': parameters_max.laser_dt,
                'observations': laser_data,
                'forward_means': forward_means,
                'forward_covariances': forward_covariances,
                'cross_entropy': cross_entropy,
                'delta_ys': delta_ys,
            }, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
