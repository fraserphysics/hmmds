"""fft_statistic_plots.py.  Make plots of statistics for psd -> threshold and pass1

python fft_statistic_plots.py statistics.pkl

Plots:

    1a. PSDs of the unfiltered heart rate average of a records and c
      records.
    1b. Coefficients for threshold and pass1

    2a. For each record, pass1 vs threshold

    2b. For each record, best threshold vs fit threshold
"""

import sys
import argparse
import pickle

import pint
import numpy
import numpy.linalg

import utilities
import plotscripts.utilities

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Plots of statistics')
    parser.add_argument('pickle_threshold',
                        type=str,
                        help='Path to statistics for threshold')
    parser.add_argument('pickle_pass1',
                        type=str,
                        help='Path to statistics for pass1')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--fig_path', type=str, help="path to figure")
    utilities.common_arguments(parser)
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def average_psd(psds, names):
    psd_list = [psds[name] for name in names]
    return sum(psd_list) / len(psd_list)


def plot_coefficients(axes, threshold_statistics, pass1_statistics):
    """Plot coefficients for threshold and pass1.
    """

    def _plot(statistics, label, key, color, factor):
        axes.plot(statistics['frequencies'],
                  statistics[key].reshape(-1) * factor,
                  label=label,
                  color=color,
                  linewidth=1,
                  linestyle='solid')

    _plot(threshold_statistics, 'threshold/100', 'threshold_coefficients', 'g',
          0.01)
    _plot(pass1_statistics, 'pass1', 'pass1_coefficients', 'k', 1.0)
    axes.set_xlabel('frequency/cpm')
    axes.legend()


def plot_averages(axes, statistics, prefix):
    """Plot average psd for a records and for c records.
    """
    frequencies = statistics['frequencies']

    psds = dict(
        (name, numpy.log10(psd)) for name, psd in statistics['psds'].items())
    a_names = []
    c_names = []
    for name in psds.keys():
        if name[0] == 'a':
            a_names.append(name)
        if name[0] == 'c':
            c_names.append(name)
    c_names.remove('c04')  # Severe arrhythmia in c04

    for (names, label, color) in ((a_names, f'{prefix} a_avg', 'r'),
                                  (c_names, f'{prefix} c_avg', 'b')):
        axes.plot(frequencies,
                  average_psd(psds, names),
                  label=label,
                  color=color,
                  linewidth=1,
                  linestyle='solid')
    axes.set_xlabel('frequency/cpm')
    axes.set_ylabel('log_10 PSD')
    axes.legend()


def plot_thresholds(axes, statistics):
    """For each record, plot fit threshold vs best  threshold.

    For visualizing fit threshold performance

    """
    threshold_coefficients = statistics['threshold_coefficients'].reshape(-1)
    for name, psd_ in statistics['psds'].items():
        psd = numpy.log10(psd_)
        best_threshold = statistics['best_threshold'][name]
        fit_ = numpy.dot(threshold_coefficients, psd)
        fit_threshold = min(2.0, max(-2.0, fit_))
        color = {'a': 'r', 'b': 'g', 'c': 'b', 'x': 'k'}[name[0]]
        if name in statistics['args'].record_names:
            color = 'm'
        axes.plot(best_threshold,
                  fit_threshold,
                  marker=f'${name}$',
                  color=color,
                  markersize=14,
                  linestyle='None')
    axes.set_ylabel('fit threshold')
    axes.set_xlabel('best threshold')


def plot_pass1(axes,
               threshold_statistics,
               pass1_statistics,
               pass1_threshold=0.175):
    """Plot each record in (threshold_coefficients,
    pass1_coefficients) coordinates.



    For visualizing pass1 performance
    """

    pass1_coefficients = pass1_statistics['pass1_coefficients'].reshape(-1)
    threshold_coefficients = threshold_statistics[
        'threshold_coefficients'].reshape(-1)
    print('     best   fit pass1')
    for name, psd_ in threshold_statistics['psds'].items():
        threshold_psd = numpy.log10(psd_)
        pass1_psd = numpy.log10(pass1_statistics['psds'][name])
        pass1 = numpy.dot(pass1_coefficients, pass1_psd)
        fit_ = numpy.dot(threshold_coefficients, threshold_psd)
        threshold = min(2.0, max(-2.0, fit_))
        color = {'a': 'r', 'b': 'g', 'c': 'b', 'x': 'k'}[name[0]]
        axes.plot(pass1,
                  threshold,
                  marker=f'${name}$',
                  color=color,
                  markersize=14,
                  linestyle='None')
        best_threshold = threshold_statistics['best_threshold'][name]
        print(
            f'{name} {best_threshold:5.2f} {threshold:5.2f} {pass1-pass1_threshold:5.2f}'
        )
    axes.plot([pass1_threshold, pass1_threshold], [-2.1, 2.1], color='k')
    axes.set_ylabel('fit threshold')
    axes.set_xlabel('pass1 statistic')


def main(argv=None):
    """Call plotting functions for statistics

    """

    if argv is None:
        argv = sys.argv[1:]
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    with open(args.pickle_threshold, 'rb') as _file:
        threshold_statistics = pickle.load(_file)

    with open(args.pickle_pass1, 'rb') as _file:
        pass1_statistics = pickle.load(_file)

    fig, axeses = pyplot.subplots(nrows=2, ncols=2, figsize=(6, 8))

    plot_averages(axeses[0, 0], pass1_statistics, 'pass1')
    plot_averages(axeses[0, 0], threshold_statistics, 'threshold')
    plot_coefficients(axeses[1, 0], threshold_statistics, pass1_statistics)
    plot_pass1(axeses[0, 1], threshold_statistics, pass1_statistics)
    plot_thresholds(axeses[1, 1], threshold_statistics)

    fig.tight_layout()
    if args.show:
        pyplot.show()
    if args.fig_path:
        fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
