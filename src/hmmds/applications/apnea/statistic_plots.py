"""statistic_plots.py.  Make plots for choosing statistics for pass1.

python statistic_plots.py --show foo.pdf

Plot 1. Plot the PSD of the unfiltered heart rate.

"""

import sys
import argparse
import pickle

import pint
import numpy
import numpy.linalg
import scipy.signal

import utilities
import plotscripts.utilities
import hmm.base

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Plots for choosing statistics')
    parser.add_argument('--A_names',
                        type=str,
                        nargs='+',
                        default=[f'a{x:02d}' for x in range(1, 21)])
    parser.add_argument('--C_names',
                        type=str,
                        nargs='+',
                        default=[f'c{x:02d}' for x in range(1, 11)])
    parser.add_argument('--X_names',
                        type=str,
                        nargs='+',
                        default=[f'x{x:02d}' for x in range(1, 36)])
    parser.add_argument('--B_names',
                        type=str,
                        nargs='+',
                        default=[f'b{x:02d}' for x in range(1, 5)])
    parser.add_argument('--sample_rate_in',
                        type=int,
                        default=2,
                        help='Samples per second of input')
    parser.add_argument('--data_dir',
                        type=str,
                        default='../../../../build/derived_data/ECG/',
                        help='Path to heart rate data for reading')
    parser.add_argument(
        '--model',
        type=str,
        default='../../../../build/derived_data/apnea/models/two_ar5_masked',
        help='Path to model')
    parser.add_argument('--format',
                        type=str,
                        default='{0}/{1}_self_AR3/heart_rate',
                        help='Map from (data_dir,name) to file of heart_rates')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('fig_path', type=str, help="path to figure")
    utilities.common_arguments(parser)
    args = parser.parse_args(argv)
    args.sample_rate_in *= PINT('Hz')
    utilities.join_common(args)
    return args


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    trouble = 'a11 c10'.split()
    colors = ['r', 'b']
    fig, axeses = pyplot.subplots(nrows=2, ncols=2, figsize=(6, 4))
    fig.tight_layout()

    records = dict(
        (name, utilities.Pass1(name, args))
        for name in args.A_names + args.C_names + args.X_names + args.B_names)
    frequencies = records[args.A_names[0]].frequencies

    def plot_psd(names, label, color):
        psd_list = [records[name].psd for name in names]
        for name in names:
            assert numpy.array_equal(frequencies, records[name].frequencies)
        psd_average = sum(psd_list) / len(psd_list)
        for axes in axeses[0, :2]:
            axes.plot(frequencies,
                      numpy.log(psd_average),
                      label=label,
                      color=color,
                      linewidth=3,
                      linestyle='solid')
        #for psd in psd_list:
        #    axeses[0].plot(frequencies, numpy.log(psd), color=color, linestyle='dotted')

    plot_psd(args.A_names, r'$\bar a$', 'r')
    plot_psd(args.C_names, r'$\bar c$', 'b')
    for axes in axeses[0, :2]:
        for name, color in zip(trouble, colors):
            axes.plot(frequencies,
                      numpy.log(records[name].psd),
                      color=color,
                      linestyle='dotted',
                      label=name)
        axes.set_xlabel('frequency/cpm')
        axes.set_ylabel('log psd')

    axeses[0, 1].set_xlim(0.5, 4.0)
    axeses[0, 1].set_ylim(0, 3.5)
    for axes in axeses[0, :]:
        axes.legend()

    def plot_stats(axes, names, color):
        for name in names:
            record = records[name]
            axes.plot(record.statistic_1(),
                      record.statistic_2(),
                      color=color,
                      marker=f'${name}$',
                      markersize=14,
                      linestyle='None')

    for axes in axeses[1, :]:
        plot_stats(axes, args.A_names, 'r')
        plot_stats(axes, args.C_names, 'b')
        plot_stats(axes, args.B_names, 'g')
        plot_stats(axes, args.X_names, 'k')
        axes.set_xlabel(r'$F(PSD)$')
        axes.set_ylabel(r'$G(PSD)$')
        y = numpy.linspace(0, 5000, 2)
        x = .36 - y * 0.
        axes.plot(x, y, linestyle='dotted')
    axeses[1, 1].set_xlim(0.307, 0.417)
    axeses[1, 1].set_ylim(200, 1500)
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
