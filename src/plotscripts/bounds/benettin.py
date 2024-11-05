""" benettin_plot.py Make figure to illustrate calculating Lyapunov exponents.
"""

import sys
import argparse
import pickle
import math

import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Figure to illustrate calculating Lyapunov exponents')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--n_traces',
                        type=int,
                        default=3,
                        help='number of example traces to plot')
    parser.add_argument('data', type=str, help='Path to input data')
    parser.add_argument('figure', type=str, help='Path to result')
    return parser.parse_args(argv)


def main(argv=None):
    """Make time series picture with fine, coarse, filtered data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    figure, (top_axes, bottom_axes) = pyplot.subplots(nrows=2,
                                                      figsize=(6, 6),
                                                      sharex=True)

    # Read and unpack data
    with open(args.data, 'rb') as _file:
        _dict = pickle.load(_file)
    r_run_time = _dict['r_run_time']
    _args = _dict['args']
    time_step = _args.time_step

    n_runs, n_times, three = r_run_time.shape
    assert three == 3
    divisors = numpy.arange(1.0, n_times + .5, 1.0)
    times = numpy.linspace(0, _args.t_run, n_times)

    augment = _args.dev_state / _args.grid_size
    five_percent = int(math.floor(0.05 * n_runs))
    ninety_five_percent = int(math.ceil(0.95 * n_runs))

    def calculate_and_plot(axes, r_values):
        """
        """
        log_r = numpy.log(r_values)
        cumsum = numpy.cumsum(log_r, axis=1) / time_step

        _sorted = numpy.sort(cumsum, axis=0)
        axes.plot(times,
                  _sorted[five_percent, :, 0] / divisors,
                  label=r'5\%',
                  color='k',
                  linewidth=2)
        for n_run in range(min(n_runs, args.n_traces)):
            axes.plot(times,
                      cumsum[n_run, :, 0] / divisors,
                      label=f'sample {n_run}')
        axes.plot(times,
                  _sorted[ninety_five_percent, :, 0] / divisors,
                  label=r'95\%',
                  color='k',
                  linewidth=2)
        axes.set_ylabel(r'$\hat \lambda (t)$')
        axes.set_ylim(0.5, 1.5)
        # Move yaxis to the right hand side so that the difference
        # between the top and bottom is easy to see
        axes.yaxis.set_label_position("right")
        axes.tick_params(labelleft=False, labelright=True)
        axes.tick_params(bottom=True, top=True, left=True, right=True)
        axes.legend()

    calculate_and_plot(top_axes, r_run_time)
    calculate_and_plot(bottom_axes, r_run_time + augment)
    bottom_axes.set_xlabel(r'$\tau$')

    if args.show:
        pyplot.show()
    figure.savefig(args.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
