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

import plotscripts.introduction.laser


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
    parser.add_argument('parameters_in', type=str, help='path to file')
    parser.add_argument('parameters_out', type=str, help='path to file')
    parser.add_argument('--plot_data', type=str, help='Path to store data')
    return parser.parse_args(argv)


def objective_function(values_in, laser_data, n_particles, rng,
                       parameter_class):
    """For optimization"""
    parameter = parameter_class(*values_in)
    lorenz_system = hmmds.applications.laser.particle.make_lorenz_system(
        parameter, rng)
    numpy.seterr(divide='raise')
    try:
        particles, forward_means, forward_covariances, log_likelihood = lorenz_system.forward_filter(
            laser_data, n_particles, threshold=0.5)
    except FloatingPointError:
        log_likelihood = -1e6
        print('caught ZeroDivisionError')
    print(f"""objective_function = {log_likelihood}""")

    return -log_likelihood


# Powell, BFGS, Nelder-Mead
def optimize(initial_parameters,
             laser_data,
             n_particles,
             rng,
             method='Powell',
             options={}):

    defaults = initial_parameters.values()
    result = scipy.optimize.minimize(
        objective_function,
        defaults,
        method=method,
        options=options,
        args=(laser_data, n_particles, rng,
              hmmds.applications.laser.optimize_ekf.Parameters),
    )
    parameters_max = hmmds.applications.laser.optimize_ekf.Parameters(*result.x)
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

    if args.parameter_type == 'GUI_out':
        parameters = hmmds.applications.laser.optimize_ekf.explore_to_parameters(
            args.parameters_in)
    elif args.parameter_type == 'parameter':
        parameters = hmmds.applications.laser.optimize_ekf.read_parameters(
            args.parameters_in)
    else:
        raise RuntimeError(
            f'parameter_type {args.parameter_type} not recognized')

    laser_data_y_t = plotscripts.introduction.laser.read_data(args.laser_data)
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
    parameters_max.write(args.parameters_out)
    if args.method != 'skip':
        with open(args.parameters_out, 'a') as _file:
            _file.write(f"""f_max {-result.fun}
success {result.success}
iterations {result.nit}
message {result.message}
n_data {args.length}""")

    if args.plot_data is None:
        return 0
    lorenz_system = hmmds.applications.laser.particle.make_lorenz_system(
        parameters_max, rng)
    particles, forward_means, forward_covariances, log_likelihood = lorenz_system.forward_filter(
        laser_data, args.n_particles, threshold=0.5)

    with open(args.plot_data, 'wb') as _file:
        pickle.dump(
            {
                'dt': parameters_max.laser_dt,
                'observations': laser_data,
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
