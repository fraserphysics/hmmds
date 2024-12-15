"""plot_like_lor.py options <data>.  Make figure of cross entropy vs number of states

"""

import sys
import argparse
import pickle

import numpy

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Plot cross entropy vs number of states')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('like_lor_data', type=str, help='Path to data')
    parser.add_argument('benettin_data', type=str, help='Path to data')
    parser.add_argument('figure_path', type=str, help='Path to result')
    return parser.parse_args(argv)


def unpack_data(data: dict):
    """
    Args:
        data:
    """
    n_test = data['args'].n_test
    n_states2log_likelihood = {}
    for key in data.keys():
        if key == 'args':
            continue
        n_states = data[key]['n_states']
        log_likelihood = data[key]['log_likelihood']
        n_states2log_likelihood[n_states] = log_likelihood
    n_pairs = len(n_states2log_likelihood)
    n_states = numpy.empty(n_pairs)
    log_likelihood = numpy.empty(n_pairs)
    for i, key in enumerate(sorted(n_states2log_likelihood.keys())):
        n_states[i] = key
        log_likelihood[i] = n_states2log_likelihood[key]
    return n_test, n_states, log_likelihood


def main(argv=None):
    """ Plot cross entropy vs number of states

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    figure, axes = pyplot.subplots(figsize=(6, 4))

    with open(args.benettin_data, 'rb') as file_:
        lyapunov_exponent = pickle.load(file_)['spectrum'][0]
    with open(args.like_lor_data, 'rb') as file_:
        data = pickle.load(file_)

    n_test, n_states, log_likelihood = unpack_data(data)

    cross_entropy = -log_likelihood / n_test
    axes.semilogx(n_states, cross_entropy, label='$\hat h$')
    axes.set_xlabel(r'$n_{\rm{states}}$')
    axes.set_ylabel(r'$\hat h/\rm{nats}$')

    limit = numpy.ones(
        len(n_states)) * lyapunov_exponent * data['args'].t_sample
    axes.semilogx(n_states, limit, label=r'$\lambda$')
    min_y = 0
    max_y = 0.9
    axes.set_ylim(min_y, max_y)

    step2 = .25
    axes.set_yticks(
        numpy.arange(step2 * round(min_y / step2), step2 * round(max_y / step2),
                     step2))

    min_y, max_y = axes.get_ylim()
    min_x, max_x = axes.get_xlim()
    ax2 = axes.twinx()
    ax2.set_xlim(min_x, max_x)
    min_y2 = min_y / numpy.log(2)
    max_y2 = max_y / numpy.log(2)
    ax2.set_ylim(min_y2, max_y2)
    ax2.set_yticks(
        numpy.arange(step2 * round(min_y2 / step2),
                     step2 * round(max_y2 / step2), step2))
    ax2.set_ylabel(r'$\hat h/\rm{bits}$')
    axes.legend()

    figure.subplots_adjust(bottom=0.15)  # Make more space for label

    if args.show:
        pyplot.show()
    figure.savefig(args.figure_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
