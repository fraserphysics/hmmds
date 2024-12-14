"""entropy_particle.py Estimate cross-entropy from run of particle filter.

python entropy_particle.py input_path

"""

import sys
import argparse
import pickle

import numpy
import numpy.linalg
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as pyplot


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Estimate entropy')
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
    log_gamma = numpy.log(gamma)
    cum_sum = numpy.cumsum(log_gamma)
    entropy = -cum_sum / numpy.arange(1, len(cum_sum) + 1)

    figure, axes = pyplot.subplots(nrows=1, ncols=1)

    axes.plot(entropy[50:] / .15, label='$\hat h$')
    x_max = len(entropy[50:])
    y_level = 2 * 0.906 / 3
    axes.plot([0, x_max], [y_level, y_level], label=r'$\frac{2}{3}\lambda$')
    axes.set_ylabel('nats')
    axes.set_xlabel('n_samples')
    axes.legend()
    #print(f'{entropy[-1]/0.1=}')
    #pyplot.show()
    figure.savefig(args.fig_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
