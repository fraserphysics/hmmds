"""figure_data.py: Write data for figures in the book

"""

import sys
import argparse
import pickle
import copy

import numpy

import optimize_ekf
import hmmds.synthetic.filter.lorenz_sde
import hmmds.applications.laser.utilities


# Function names match figures in the book.  pylint: disable = invalid-name
def parse_args(argv):
    """Command line:

    python figure_data.py --LaserLogLike path laser_data parameters """
    parser = argparse.ArgumentParser(
        description='Write data for figures in book about laser')
    parser.add_argument('--LaserLP5',
                        type=str,
                        help='Write simulated laser data here')
    parser.add_argument(
        '--LaserLogLike',
        type=str,
        help='Write dependence of log likelihood on s and b here')
    parser.add_argument('--LaserStates',
                        type=str,
                        help='Write 250 states from particle filter here')
    parser.add_argument('--LaserForecast',
                        type=str,
                        help='Write sequence of 400 values here')
    parser.add_argument('--LaserHist',
                        type=str,
                        help='Write 600 values for histogram here')
    parser.add_argument('--parameters', type=str, help='Path of parameter file')
    parser.add_argument('--laser_data', type=str, help='path of data file')
    return parser.parse_args(argv)


Tasks = {}  # Keys are function names.  Values are functions.


def register(func):
    """Decorator that puts function in Tasks dictionary
    """
    Tasks[func.__name__] = func
    return func


def simulate(parameters, n_samples):
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
            samples[i - 1], 0, t_sample, parameters.s, parameters.r,
            parameters.b)
    return samples


def observe(parameters, n_samples):
    """Simulate noiseless observations of noiseless dynamics.
    """
    states = simulate(parameters, n_samples)
    return (parameters.x_ratio * states[:, 0]**2 +
            parameters.offset).astype(int)


@register
def LaserLP5(args):
    """Return simulated laser data over fit range.
    """
    return {'250_simulated_observations': observe(args.parameters, 250)}


@register
def LaserLogLike(args):
    """Dependence of Log Likelihood on s and b
    """
    s = numpy.linspace(7.0, 10.0, 7, endpoint=True)
    b = numpy.linspace(2.35, 2.6, 6, endpoint=True)
    log_likelihood = numpy.empty((len(s), len(b)))
    for i, s_ in enumerate(s):
        for j, b_ in enumerate(b):
            parameters = copy.deepcopy(args.parameters)
            parameters.s = s_
            parameters.b = b_
            sde, initial_distribution, _ = optimize_ekf.make_non_stationary(
                parameters, None)
            log_likelihood[i, j] = sde.log_likelihood(initial_distribution,
                                                      args.laser_data)
    return {'s': s, 'b': b, 'log_likelihood': log_likelihood}


@register
def LaserStates(args):
    """Sequence of 250 states from particle filter
    """
    sde, initial_distribution, _ = optimize_ekf.make_non_stationary(
        args.parameters, None)
    forward_means, _ = sde.forward_filter(initial_distribution, args.laser_data)
    return {'forward_means': forward_means}


@register
def LaserForecast(args):
    """Return 400 simulated laser data points (150 beyond fit range).

    FixMe: Should run filter to estimate state at 250 and then forecast.
    """
    return {'400_simulated_observations': observe(args.parameters, 400)}


@register
def LaserHist(args):
    """Make a histogram of the first 600 samples.
    """
    count = numpy.zeros(256, numpy.int32)
    for y in args.laser_data[0:600]:
        count[y] += 1
    return {'count': count}


def main(argv=None):
    """ Write data for figures in the book
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]
    args = parse_args(argv)

    if args.parameters:
        args.parameters = hmmds.applications.laser.utilities.read_parameters(
            args.parameters)
    if args.laser_data:
        laser_data_y_t = hmmds.applications.laser.utilities.read_tang(
            args.laser_data)
        assert laser_data_y_t.shape == (2, 2876)
        args.laser_data = laser_data_y_t[1, :].astype(int).reshape((2876, 1))

    for name, function in Tasks.items():
        path = getattr(args, name)
        if path is not None:
            with open(path, 'wb') as _file:
                pickle.dump(function(args), _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
