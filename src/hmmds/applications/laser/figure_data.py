"""figure_data.py: Write data for figures in the book

"""

import sys
import argparse
import pickle
import copy

import numpy

import hmmds.synthetic.filter.lorenz_sde
import hmmds.applications.laser.utilities


# Function names match figures in the book.  pylint: disable = invalid-name
def parse_args(argv):
    """Command line:

    python figure_data.py --LaserLogLike path laser_data parameters """
    parser = argparse.ArgumentParser(
        description='Write data for figures in book about laser')
    parser.add_argument('--TrainingInterval',
                        type=int,
                        nargs=2,
                        default=[0, 250],
                        help='Segment of laser data')
    parser.add_argument('--TestingInterval',
                        type=int,
                        nargs=2,
                        default=[250, 500],
                        help='Segment of laser data')
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


@register
def LaserLP5(args):
    """Return simulated laser data over fit range.
    """
    t_start, t_stop = args.TrainingInterval
    return {
        't_start':
            t_start,
        't_stop':
            t_stop,
        'training_simulated_observations':
            hmmds.applications.laser.utilities.observe(args.parameters, t_stop),
        'training_laser_data':
            args.laser_data[t_start:t_stop]
    }


@register
def LaserLogLike(args):
    """Dependence of Log Likelihood on s and b
    """
    t_start, t_stop = args.TrainingInterval
    training_data = args.laser_data[t_start:t_stop]
    print(f'{training_data.shape=}')
    s_center = args.parameters.s
    b_center = args.parameters.b
    s = numpy.linspace(s_center * .9, s_center * 1.1, 7, endpoint=True)
    b = numpy.linspace(b_center * .95, b_center * 1.05, 6, endpoint=True)
    log_likelihood = numpy.empty((len(s), len(b)))
    for i, s_ in enumerate(s):
        for j, b_ in enumerate(b):
            parameters = copy.deepcopy(args.parameters)
            parameters.s = s_
            parameters.b = b_
            sde, initial_distribution, _ = hmmds.applications.laser.utilities.make_non_stationary(
                parameters, None)
            log_likelihood[i, j] = sde.log_likelihood(initial_distribution,
                                                      training_data)
    return {
        't_start': t_start,
        't_stop': t_stop,
        's': s,
        'b': b,
        'log_likelihood': log_likelihood
    }


@register
def LaserStates(args):
    """Sequence of states from particle filter
    """
    t_start, t_stop = args.TrainingInterval
    sde, initial_distribution, _ = hmmds.applications.laser.utilities.make_non_stationary(
        args.parameters, None)
    forward_means, _ = sde.forward_filter(initial_distribution,
                                          args.laser_data[t_start:t_stop])
    return {
        't_start': t_start,
        't_stop': t_stop,
        'forward_means': forward_means
    }


@register
def LaserForecast(args):
    """Return simulated laser data points over the testing range.

    FixMe: Should run filter to estimate state at 250 and then forecast.
    """
    t_start, t_stop = args.TestingInterval
    initial_state = LaserStates(args)['forward_means'][-1]
    args.parameters.set_initial_state(initial_state)
    return {
        't_start':
            t_start,
        't_stop':
            t_stop,
        'next_data':
            args.laser_data[t_start:t_stop],
        'forecast_observations':
            hmmds.applications.laser.utilities.observe(args.parameters,
                                                       t_stop + 1 - t_start)[1:]
    }


@register
def LaserHist(args):
    """Make a histogram of the first 600 samples.
    """
    n_bins = 256
    t_start, t_stop = (0, 600)
    count = numpy.zeros(n_bins, numpy.int32)
    for y in args.laser_data[t_start:t_stop]:
        count[y] += 1
    return {
        'n_bins': n_bins,
        't_start': t_start,
        't_stop': t_stop,
        'count': count
    }


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
