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

from hmmds.applications.laser import explore
import hmmds.applications.laser.utilities


def parse_args(argv):
    """Parse the command line
    """
    # Like optimze_particle.  pylint: disable = duplicate-code
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
    parser.add_argument('--objective_function',
                        type=str,
                        default='likelihood',
                        help='l2 or likelihood')
    parser.add_argument('--length',
                        type=int,
                        default=2876,
                        help='optimize over this number of data samples')
    # FixMe: Write new module instead of using --skip
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
    with open(in_path, 'r', encoding='utf-8') as file_:
        for line in file_.readlines():
            name, value_str = line.split()
            in_dict[name] = float(value_str)
    # pylint: disable = invalid-name
    s = in_dict['s']
    r = in_dict['r']
    b = in_dict['b']
    fixed_point = hmmds.applications.laser.utilities.FixedPoint(r, s, b)
    # No type info for lorenz_sde pylint: disable = c-extension-no-member
    initial_state = hmmds.synthetic.filter.lorenz_sde.lorenz_integrate(
        fixed_point.initial_state(in_dict['delta_x']), 0.0, in_dict['delta_t'],
        s, r, b)
    parameters = hmmds.applications.laser.utilities.Parameters(
        s,
        r,
        b,
        *initial_state,
        in_dict['t_ratio'],
        in_dict['x_ratio'],
        in_dict['offset'],
    )
    return parameters


def likelihood_objective(values_in, laser_data, parameter_class):
    """Log likelihood of EKF for optimization"""
    numpy.seterr(all='raise')
    parameter = parameter_class(*values_in)
    non_stationary, initial_distribution, _ = hmmds.applications.laser.utilities.make_non_stationary(
        parameter, None)
    try:
        result = non_stationary.log_likelihood(initial_distribution, laser_data)
    except FloatingPointError:
        result = -1e100
    print(f"""objective_function = {result}""")

    return -result


def l2_objective(values_in, laser_data, parameter_class):
    """L2 difference between data and simulation for optimization"""
    parameters = parameter_class(*values_in)
    simulation = hmmds.applications.laser.utilities.observe(
        parameters, len(laser_data))
    difference = laser_data[:, 0] - simulation
    result = numpy.sqrt(difference * difference).sum()
    print(f"""objective_function = {result}""")

    return result


# Powell, BFGS, Nelder-Mead
def optimize(initial_parameters,
             laser_data,
             objective,
             method='Powell',
             options=None):
    """Call scipy.optimize.minimize.
    """

    # Like optimize_particle. pylint: disable = duplicate-code
    defaults = initial_parameters.values()
    result = scipy.optimize.minimize(
        objective,
        defaults,
        method=method,
        options=options,
        args=(laser_data, hmmds.applications.laser.utilities.Parameters),
    )
    parameters_max = hmmds.applications.laser.utilities.Parameters(*result.x)
    print(f"""
parameters_max:
{parameters_max}
f_max={-result.fun}
success={result.success}
message={result.message}
iterations={result.nit}""")
    return parameters_max, result


def main(argv=None):
    """Fit parameters for EKF to laser data.
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]
    args = parse_args(argv)

    if not args.plot_data and len(args.parameters_in_out) != 2:
        raise RuntimeError('No file specified for results')
    if len(args.parameters_in_out) > 2:
        raise RuntimeError('More than 2 positional arguments')

    # pylint: disable = duplicate-code
    if args.parameter_type == 'GUI_out':
        parameters = explore_to_parameters(args.parameters_in_out[0])
    elif args.parameter_type == 'parameter':
        parameters = hmmds.applications.laser.utilities.read_parameters(
            args.parameters_in_out[0])
    else:
        raise RuntimeError(
            f'parameter_type {args.parameter_type} not recognized')

    laser_data_y_t = hmmds.applications.laser.utilities.read_tang(
        args.laser_data)
    assert laser_data_y_t.shape == (2, 2876)
    # Put y values in global
    laser_data = laser_data_y_t[1, :].astype(int).reshape((2876, 1))

    if args.method != 'skip':
        options = {'maxiter': 20000}
        parameters_max, result = optimize(parameters,
                                          laser_data[:args.length],
                                          {
                                              'l2': l2_objective,
                                              'likelihood': likelihood_objective
                                          }[args.objective_function],
                                          method=args.method,
                                          options=options)
    else:
        parameters_max = parameters
    if len(args.parameters_in_out) == 2:
        parameters_max.write(args.parameters_in_out[1])
    if args.method != 'skip' and len(args.parameters_in_out) == 2:
        with open(args.parameters_in_out[1], 'a', encoding='utf-8') as _file:
            _file.write(f"""f_max {-result.fun}
success {result.success}
iterations {result.nit}
message {result.message}
n_data {args.length}""")

    if args.plot_data is None:
        return 0
    sde, initial_distribution, _ = hmmds.applications.laser.utilities.make_non_stationary(
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
