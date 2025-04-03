"""entropy_particle.py Estimate cross-entropy from run of particle filter.

python entropy_particle.py --dict_template study_threshold/{0}/dict.pkl 1e-2 1e-3 1e-4

"""

import sys
import os
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
    parser.add_argument('--plot_counts', action='store_true')
    parser.add_argument('--dir_template',
                        type=str,
                        default='study_threshold/{0}',
                        help='map from key to dir')
    parser.add_argument('keys',
                        type=str,
                        nargs='+',
                        help='variable part of path')
    args = parser.parse_args(argv)
    return args


def plot_key(args, axeses, key):
    """Plot data from args.dir_template(key) on axeses

    Args:
        axeses:
        dict_template: EG, study_threshold/{0}/dict.pkl
        keys: EG, e-2 1e-3 1e-4
    """
    if args.plot_counts:
        n_rows = 3
    else:
        n_rows = 2

    with open(os.path.join(args.dir_template.format(key), 'dict.pkl'),
              'rb') as file_:
        dict_in = pickle.load(file_)
    gamma = dict_in['gamma']
    offset = 50  # First resample from 200,000 to 20,000 with some
    # nice parameters
    log_gamma = numpy.log(gamma)[offset:]
    cum_sum = numpy.cumsum(log_gamma)
    entropy = -cum_sum / numpy.arange(1, len(cum_sum) + 1) / 0.15
    reference = numpy.ones(len(entropy)) * 0.906
    x = numpy.arange(offset, len(gamma))
    axeses[n_rows - 2].plot(numpy.log10(gamma), label=f'{key}')
    axeses[n_rows - 2].set_ylabel(r'log$_{10}(P(y[t]|y[0:t]))$')
    axeses[n_rows - 1].plot(x, entropy, label=f'{key}')
    axeses[n_rows - 1].plot(x, reference)
    axeses[n_rows - 1].set_xlabel(r'$t$')
    axeses[n_rows - 1].set_ylabel(r'$\hat h$')
    h_hat = entropy[-1]
    print(f'h_hat({key})= {h_hat} {(h_hat/0.906-1)*100:4.2f}%')

    if not args.plot_counts:
        return
    n_forecast = numpy.zeros(len(gamma), dtype=int)
    n_update = numpy.zeros(len(gamma), dtype=int)
    with open(os.path.join(args.dir_template.format(key), 'states_boxes.npy'),
              'rb') as file_:
        for n in range(len(gamma)):
            try:
                n_forecast[n] = len(numpy.load(file_))
                n_update[n] = len(numpy.load(file_))
            except:
                break
    axeses[n_rows - 3].plot(n_forecast, label=f'n_forecast({key})')
    axeses[n_rows - 3].plot(n_update, label=f'n_update({key})')


def main(argv=None):
    """Plot data from particle filter simulations for entropy estimation
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    if args.plot_counts:
        n_rows = 3
    else:
        n_rows = 2
    figure, axeses = pyplot.subplots(nrows=n_rows, ncols=1, sharex=True)
    for key in args.keys:
        plot_key(args, axeses, key)

    axeses[n_rows - 2].set_ylabel(r'log$_{10}(P(y[t]|y[0:t]))$')
    axeses[n_rows - 2].legend()

    axeses[n_rows - 1].set_xlabel(r'$t$')
    axeses[n_rows - 1].set_ylabel(r'$\hat h$')
    axeses[n_rows - 1].legend()

    if args.plot_counts:
        axeses[n_rows - 3].set_ylabel(r'N')
        axeses[n_rows - 3].legend()

    pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
