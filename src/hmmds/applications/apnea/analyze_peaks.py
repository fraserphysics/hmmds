"""analyze_peaks.py Characterize distribution of peak prominences in
the heart rate signal

python analyze_peaks.py --show characteristics.pkl

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
    parser.add_argument('--figure_path', type=str, help='path of file to write')
    parser.add_argument('pickle',
                        type=str,
                        help='Saved characterstics of peaks')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def plot_record(args, record_name, boundaries, axes):
    raw_dict = utilities.read_slow_class_peak(args, boundaries, record_name)
    axes.plot(raw_dict['slow'] / 10, label='slow')
    axes.plot(raw_dict['peak'], label='bin')
    axes.plot(raw_dict['class'], label='class')


def analyze_record(args, record_name, peak_dict, characteristics):
    '''Put prominences of detected peaks in the right value of peak_dict

    Args:
        args: Command line arguments
        record_name: eg 'a03'
        peak_dict: keys (0,1) values are lists of prominences
        characteristics: Include min_prominence

    '''
    min_prominence = characteristics['min_prominence']
    raw_dict = utilities.read_slow_class(args, record_name)
    peak_times, properties = utilities.peaks(raw_dict['slow'],
                                             args.heart_rate_sample_frequency,
                                             min_prominence)
    for time, prominence in zip(peak_times, properties['prominences']):
        class_ = raw_dict['class'][time]
        peak_dict[class_].append(prominence)


def plot_cdf(axes, data, color=None, label=None):
    '''Plot cumulative distribution function

    Args:
       axes: A pyplot axes instance
       data: Sorted 1-d array of floats
    '''
    y = numpy.linspace(0, 1, len(data))
    axes.plot(data, y, color=color, linewidth=2, linestyle='solid', label=label)


def plot_n(x, boundaries, axes, marks=False, label=None):
    '''Mark boundaries on plot of cumulative number

    Args:
       x: sorted 1-d array of floats
       boundaries: Boundaries of bins
       axes: A pyplot axes instance
    '''
    x_b = numpy.searchsorted(x, boundaries)
    axes.plot(boundaries, x_b, linestyle='', marker='x')
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

    with open(args.pickle, 'rb') as _file:
        characteristics = pickle.load(_file)
    boundaries = characteristics['boundaries']
    # FixMe: Decide how to divide things betweek characteristics and
    # args
    args.min_prominence = characteristics['min_prominence']

    # Find peaks
    peak_dict = {0: [], 1: []}
    for record_name in args.a_names:
        analyze_record(args, record_name, peak_dict, characteristics)

    # Sort the peaks and find indices of boundaries
    indices = {}
    for class_ in (0, 1):
        data = numpy.array(peak_dict[class_])
        data.sort()
        indices[class_] = numpy.searchsorted(data, boundaries)
        peak_dict[class_] = data

    plot_cdf(axeses[0], peak_dict[0], label='Normal')
    plot_cdf(axeses[0], peak_dict[1], label='Apnea')
    axeses[0].plot(boundaries,
                   indices[1] / len(peak_dict[1]),
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
    if args.figure_path:
        fig.savefig(args.figure_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
