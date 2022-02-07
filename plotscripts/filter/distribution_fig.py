r"""distibution_fig.py Make plot showing how well random data fits the
formula in linear_map_simulation.py for calculating the stationary
distribution.

"""

import sys
import argparse
import pickle

import numpy
import scipy.stats

import plotscripts.utilities


def parse_args(argv):
    """Parse the command line.
    """

    parser = argparse.ArgumentParser(
        description=
        'Characterize the distribution of states of a stochastic system.')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data', type=str, help='Path to data')
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Writes time series to files specified by options --xyzfile,
    --quantfile, and or --TSintro.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    with open(args.data, 'rb') as _file:
        data = pickle.load(_file)

    quantiles, sorted_x_0 = scipy.stats.probplot(
        data['x_0'],
        dist='norm',
        sparams=(0.0, data['std_deviation']),
        fit=False)

    fig, (axis_a, axis_b) = pyplot.subplots(nrows=2, figsize=(6, 8))

    axis_a.plot(numpy.array(range(len(data['x_0']))) * data['dt'],
                data['x_0'],
                marker='.',
                linestyle='None')
    axis_a.set_xlabel('t')
    axis_a.set_ylabel('$x_0$')

    axis_b.plot(quantiles, sorted_x_0, marker='.', linestyle='None')
    ends = [quantiles[0], quantiles[-1]]
    axis_b.plot(ends, ends)
    axis_b.set_xlabel('Theoretical quantiles')
    axis_b.set_ylabel('$x_0$ quantiles')
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
