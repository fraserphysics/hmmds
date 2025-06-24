"""entropy_particle.py Estimate cross-entropy from run of particle filter.

python entropy_particle.py --dir_template r_threshold/{0}

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

    parser = argparse.ArgumentParser(description='''Estimate entropy, eg:
python entropy_particle.py --dir_template study_threshold5k/{0} 1e-2 3e-3 1e-3'''
                                    )
    parser.add_argument('--plot_counts', action='store_true')
    parser.add_argument('--plot_likelihood', action='store_true')
    parser.add_argument('--show_h_hat',
                        action='store_true',
                        help='put to labels on the right hand y axis')
    parser.add_argument('--ylim', type=float, nargs=2, help='Set range of y')
    parser.add_argument('--dir_template',
                        type=str,
                        default='study_threshold5k/{0}',
                        help='map from key to dir')
    parser.add_argument('--save',
                        type=str,
                        help='path to result.  Show if not set')
    parser.add_argument('keys',
                        type=str,
                        nargs='+',
                        help='variable part of path')
    args = parser.parse_args(argv)
    return args


def plot_key(args, axes_dict, key):
    """Plot data from args.dir_template(key) on axeses

    Args:
        axeses:
        dict_template: EG, study_threshold/{0}/dict.pkl
        keys: EG, e-2 1e-3 1e-4
    """

    with open(os.path.join(args.dir_template.format(key), 'dict.pkl'),
              'rb') as file_:
        dict_in = pickle.load(file_)
    gamma = dict_in['gamma']
    offset = 50  # First resample from 200,000 to 20,000 with some
    # nice parameters
    log_gamma = numpy.log(gamma)[offset:]
    cum_sum = numpy.cumsum(log_gamma)
    entropy = -cum_sum / numpy.arange(1, len(cum_sum) + 1) / 0.15
    true_h = 0.906
    reference = numpy.ones(len(entropy)) * true_h
    x = numpy.arange(offset, len(gamma))

    if len(args.keys) == 1:
        axes_dict['entropy'].plot(x, entropy, label=r'$\hat h$')
    else:
        axes_dict['entropy'].plot(x, entropy, label=f'{key}')
    if key == args.keys[-1]:  # only plot the reference line once
        axes_dict['entropy'].plot(x, reference, label=r'$\lambda$')
    axes_dict['entropy'].set_xlabel(r'$n_{\text{samples}}$')
    axes_dict['entropy'].set_ylabel(r'$\hat h/\text{nats}$')
    h_hat = entropy[-1]
    if args.show_h_hat:
        min_y, max_y = axes_dict['entropy'].get_ylim()
        min_x, max_x = axes_dict['entropy'].get_xlim()
        ax2 = axes_dict['entropy'].twinx()
        ax2.set_xlim(min_x, max_x)
        ax2.set_ylim(min_y, max_y)
        ax2.yaxis.set_major_formatter(
            matplotlib.ticker.StrMethodFormatter("{x:.3f}"))
        ax2.set_yticks((true_h, h_hat))

    if 'likelihood' in axes_dict:
        axes_dict['likelihood'].plot(numpy.log10(gamma), label=f'{key}')
        axes_dict['likelihood'].set_ylabel(r'log$_{10}(P(y[t]|y[0:t]))$')

    if 'counts' not in axes_dict:
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
    axes_dict['counts'].plot(n_forecast, label=f'n_forecast({key})')
    axes_dict['counts'].plot(n_update, label=f'n_update({key})')
    argmin = n_update[100:].argmin() + 100
    print(
        f'Minimum of update for {key}: n_update[{argmin}]={n_update[argmin]}\n')


def main(argv=None):
    """Plot data from particle filter simulations for entropy estimation
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    n_rows = 1
    if args.plot_counts:
        n_rows += 1
    if args.plot_likelihood:
        n_rows += 1
    figure, axes_array = pyplot.subplots(nrows=n_rows, ncols=1, sharex=True)
    if n_rows == 1:
        axes_array = [axes_array]
    row_count = n_rows - 1
    axes_dict = {'entropy': axes_array[row_count]}
    if args.plot_counts:
        row_count -= 1
        axes_dict['counts'] = axes_array[row_count]
    if args.plot_likelihood:
        row_count -= 1
        axes_dict['likelihood'] = axes_array[row_count]

    for key in args.keys:
        plot_key(args, axes_dict, key)

    axes_dict['entropy'].legend()
    if args.ylim:
        print(f'{args.ylim[0]=} {args.ylim[1]=}')
        axes_dict['entropy'].set_ylim(args.ylim[0], args.ylim[1])

    if 'counts' in axes_dict:
        axes_dict['counts'].set_ylabel(r'N')
        axes_dict['counts'].legend()

    if 'likelihood' in axes_dict:
        axes_dict['likelihood'].set_ylabel(r'log$_{10}(P(y[t]|y[0:t]))$')
        axes_dict['likelihood'].legend()

    if args.save:
        figure.savefig(args.save)
    else:
        pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
