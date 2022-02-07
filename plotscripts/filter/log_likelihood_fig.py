r"""max_like.py

Check likelihood of data from the state space model defined in
linear_map_simulatinon.py

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

    parser = argparse.ArgumentParser(description='Illustrate MLE.')

    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('data', type=str)
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make a plot of log likelihood vs the state noise amplitude b

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    with open(args.data, 'rb') as _file:
        data = pickle.load(_file)

    fig, (axis_a, axis_b) = pyplot.subplots(nrows=2, figsize=(6, 8))

    axis_a.plot(data['x_data'][:, 0])
    axis_a.set_xlabel('$t$')
    axis_a.set_ylabel('$x_0$')

    axis_b.plot(data['b_array'], data['log_like'])
    axis_b.set_xlabel('$b$')
    axis_b.set_ylabel('Log Likelihood')

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
