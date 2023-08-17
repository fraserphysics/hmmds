"""analyze_peaks.py Characterize distribution of peaks in the heart rate signal

"""
import sys
import os.path
import pickle
import argparse

import numpy

import hmm.base
import hmm.simple

import utilities
import plotscripts.utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Analyze peaks of heart rate sign")
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    utilities.common_arguments(parser)
    parser.add_argument('figure_path', type=str, help='path of file to write')
    parser.add_argument('boundaries_path',
                        type=str,
                        help='path of file to write')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def plot_record(args, record_name, boundaries, axes):
    raw_dict = utilities.read_slow_class_peak(args, boundaries, record_name)
    axes.plot(raw_dict['slow'] / 10, label='slow')
    axes.plot(raw_dict['peak'], label='bin')
    axes.plot(raw_dict['class'], label='class')


def analyze_record(args, record_name, peak_dict):
    raw_dict = utilities.read_slow_class(args, record_name)
    peaks, properties = utilities.peaks(raw_dict['slow'],
                                        args.heart_rate_sample_frequency)
    for peak, prominence in zip(peaks, properties['prominences']):
        peak_dict[raw_dict['class'][peak]].append(prominence)


def plot_cdf(axes, data, color=None, label=None):
    y = numpy.linspace(0, 1, len(data))
    axes.plot(data, y, color=color, linewidth=2, linestyle='solid', label=label)
    return data


def plot_n(x, boundaries, axes, marks=False, label=None):
    x_b = numpy.searchsorted(x, boundaries[1])
    axes.plot(boundaries[1], x_b, linestyle='', marker='x')
    y = numpy.arange(len(x))
    axes.plot(x, y, label=label)
    if marks:
        axes.plot([0] * len(x_b), x_b, linestyle='', marker='x')


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    fig, axeses = pyplot.subplots(nrows=3, figsize=(6, 8))
    axeses[0].sharex(axeses[1])

    # Find peaks
    peak_dict = {0: [], 1: []}
    for record_name in args.a_names:
        analyze_record(args, record_name, peak_dict)

    # Sort the peaks
    for class_ in (0, 1):
        data = numpy.array(peak_dict[class_])
        data.sort()
        peak_dict[class_] = data

    # Set boundaries so that each apnea bin has 1,000 peaks
    boundaries = []
    for i, index in enumerate(range(0, len(peak_dict[1]), 1000)):
        boundaries.append((index / len(peak_dict[1]), peak_dict[1][index]))
    boundaries = numpy.array(boundaries).T

    plot_cdf(axeses[0], peak_dict[0], label='Normal')
    plot_cdf(axeses[0], peak_dict[1], label='Apnea')
    axeses[0].plot(boundaries[1],
                   boundaries[0],
                   linestyle='',
                   marker='x',
                   label='boundaries')
    axeses[0].set_ylabel('cdf')

    plot_n(peak_dict[0], boundaries, axeses[1], True, label='Normal')
    plot_n(peak_dict[1], boundaries, axeses[1], label='Apnea')
    axeses[1].set_ylabel('Number')
    axeses[1].set_xlabel('peak prominences')

    plot_record(args, 'a03', boundaries, axeses[2])
    for axes in axeses:
        axes.legend()
    if args.show:
        pyplot.show()
    fig.savefig(args.figure_path)
    with open(args.boundaries_path, mode='wb') as _file:
        pickle.dump(boundaries[1], _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
