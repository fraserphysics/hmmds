"""entropy_particle.py Estimate cross-entropy from run of particle filter.

python entropy_particle.py --dict_template study_threshold/{0}/dict.pkl 1e-2 1e-3 1e-4

"""

import sys
import argparse
import pickle

import numpy
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as pyplot


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Estimate entropy')
    parser.add_argument('--dict_template',
                        type=str,
                        default='study_threshold/{0}/dict.pkl',
                        help='map from key to path')
    parser.add_argument('keys', type=str, nargs='+', help='variable part of path')
    args = parser.parse_args(argv)
    return args


def plot_file(axeses, dict_template, key):
    """Plot data from dict_template(key) on axeses

    Args:
        axeses:
        dict_template: EG, study_threshold/{0}/dict.pkl
        keys: EG, e-2 1e-3 1e-4
    """
    with open(dict_template.format(key), 'rb') as file_:
        dict_in = pickle.load(file_)
    gamma = dict_in['gamma']
    offset = 50  # First resample from 200,000 to 20,000 with some
    # nice parameters
    log_gamma = numpy.log(gamma)[offset:]
    cum_sum = numpy.cumsum(log_gamma)
    entropy = -cum_sum / numpy.arange(1, len(cum_sum) + 1) / 0.15
    reference = numpy.ones(len(entropy)) * 0.906
    x = numpy.arange(offset, len(gamma))
    axeses[0].plot(numpy.log10(gamma))
    axeses[0].set_ylabel(r'log$_{10}(P(y[t]|y[0:t]))$')
    axeses[1].plot(x, entropy, label=f'{key}')
    axeses[1].plot(x, reference)
    axeses[1].set_xlabel(r'$t$')
    axeses[1].set_ylabel(r'$\hat h$')
    h_hat = entropy[-1]
    print(f'{h_hat=} {(h_hat/0.906-1)*100:4.2f}%')


def main(argv=None):
    """Plot data from particle filter simulations for entropy estimation
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    figure, axeses = pyplot.subplots(nrows=2, ncols=1, sharex=True)
    for key in args.keys:
        plot_file(axeses, args.dict_template, key)
    axeses[0].set_ylabel(r'log$_{10}(P(y[t]|y[0:t]))$')
    axeses[1].set_xlabel(r'$t$')
    axeses[1].set_ylabel(r'$\hat h$')
    axeses[1].legend()

    pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
