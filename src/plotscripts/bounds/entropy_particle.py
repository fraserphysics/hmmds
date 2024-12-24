"""entropy_particle.py Estimate cross-entropy from run of particle filter.

python entropy_particle.py input_path

"""

import sys
import argparse
import pickle

import numpy
import numpy.linalg

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Estimate entropy')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('input', type=str, help='Path to data')
    parser.add_argument('fig_path', type=str, help='Path to figure')
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    """Plot some stuff
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    with open(args.input, 'rb') as file_:
        dict_in = pickle.load(file_)
    gamma = dict_in['gamma']
    log_gamma = numpy.log(gamma)[14:]
    cum_sum = numpy.cumsum(log_gamma)
    entropy = -cum_sum / numpy.arange(1, len(cum_sum) + 1)

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    figure, axes = pyplot.subplots(figsize=(6, 4))

    y_values = entropy / 0.15
    axes.plot(y_values, label=r'$\hat h$')
    x_max = len(y_values)
    y_level = 0.906
    axes.plot([0, x_max], [y_level, y_level], label=r'$\lambda$')
    axes.set_ylabel(r'$\hat h/\rm{nats}$')
    axes.set_xlabel(r'$n_{\rm{samples}}$')
    axes.set_ylim(0, 5.99)
    axes.legend()
    if args.show:
        print(f'{y_values[-1]=}')
        pyplot.show()
    figure.savefig(args.fig_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
