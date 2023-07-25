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
    #parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def analyze_record(args, record_name, peak_dict):
    raw_dict = utilities.read_slow_class(args, record_name)
    peaks, properties = utilities.peaks(raw_dict['slow'],
                                        args.heart_rate_sample_frequency)
    for peak, prominence in zip(peaks, properties['prominences']):
        peak_dict[raw_dict['class'][peak]].append(prominence)


def plot_cdf(axes, data, color=None):
    y = numpy.linspace(0, 1, len(data))
    axes.plot(data, y, color=color, linewidth=2, linestyle='solid')
    return data


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    fig, axeses = pyplot.subplots(nrows=2, figsize=(6, 8))

    peak_dict = {0: [], 1: []}
    for record_name in args.a_names:
        analyze_record(args, record_name, peak_dict)
    for class_ in (0, 1):
        data = numpy.array(peak_dict[class_])
        data.sort()
        peak_dict[class_] = data
    boundaries = []
    for i, index in enumerate(range(0, len(peak_dict[1]), 1000)):
        boundaries.append((index / len(peak_dict[1]), peak_dict[1][index]))
    boundaries = numpy.array(boundaries).T
    print(f'{boundaries.shape=}')

    plot_cdf(axeses[0], peak_dict[0])
    plot_cdf(axeses[0], peak_dict[1])
    axeses[0].plot(boundaries[1], boundaries[0], linestyle='', marker='x')

    def foo(x, marks=False):
        x_b = numpy.searchsorted(x, boundaries[1])
        axeses[1].plot(boundaries[1], x_b, linestyle='', marker='x')
        y = numpy.arange(len(x))
        axeses[1].plot(x, y)
        if marks:
            axeses[1].plot([0] * len(x_b), x_b, linestyle='', marker='x')

    foo(peak_dict[0], True)
    foo(peak_dict[1])
    if args.show:
        pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
