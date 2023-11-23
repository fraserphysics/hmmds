"""analyze_peaks.py Characterize distribution of peak prominences in
the heart rate signal

python analyze_peaks.py --show characteristics.pkl

"""
import sys
import os.path
import pickle
import argparse

import numpy

import utilities
import plotscripts.utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Analyze peaks of heart rate sign")
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--normalize',
                        action='store_true',
                        help="Normalize heart rate signal")
    utilities.common_arguments(parser)
    parser.add_argument('--figure_path', type=str, help='path of file to write')
    parser.add_argument('config_path',
                        type=str,
                        help='Path for statistics of peaks')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def analyze_record(args, config, record_name, peak_dict):
    '''Put prominences of detected peaks in the right value of peak_dict

    Args:
        args: Command line arguments
        record_name: eg 'a03'
        peak_dict: keys (0,1) values are lists of prominences

    '''
    heart_rate = utilities.HeartRate(args, record_name, config)
    heart_rate.filter_hr()
    heart_rate.read_expert()
    heart_rate.find_peaks()

    for time, prominence in zip(heart_rate.peaks, heart_rate.peak_prominences):
        minute = int(time /
                     heart_rate.hr_sample_frequency.to('1/minute').magnitude)
        if minute >= len(heart_rate.expert):
            break
        peak_dict[heart_rate.expert[minute]].append(prominence)


def plot_cdf(axes, data, color=None, label=None):
    '''Plot cumulative distribution function

    Args:
       axes: A pyplot axes instance
       data: Sorted 1-d array of floats
    '''
    y = numpy.linspace(0, 1, len(data))
    axes.plot(data, y, color=color, linewidth=2, linestyle='solid', label=label)


def plot_n(x, axes, label=None):
    '''Mark boundaries on plot of cumulative number

    Args:
       x: sorted 1-d array of floats
       axes: A pyplot axes instance
    '''
    y = numpy.arange(len(x))
    axes.plot(x, y, label=label)


def main(argv=None):
    """Plot cdf and number of peak prominences for normal and apnea.

    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    fig, axeses = pyplot.subplots(nrows=2, figsize=(6, 8))

    axeses[0].sharex(axeses[1])
    axeses[0].set_xlim(-1, 50)

    with open(args.config_path, 'rb') as _file:
        config = pickle.load(_file)

    if args.records:
        records = args.records
    else:
        records = args.a_names

    # Find peaks
    peak_dict = {0: [], 1: []}
    for record_name in records:
        analyze_record(args, config, record_name, peak_dict)

    # Sort the peaks
    for class_ in (0, 1):
        data = numpy.array(peak_dict[class_])
        data.sort()
        peak_dict[class_] = data

    plot_cdf(axeses[0], peak_dict[0], label='Normal')
    plot_cdf(axeses[0], peak_dict[1], label='Apnea')
    axeses[0].set_ylabel('cdf')

    plot_n(peak_dict[0], axeses[1], label='Normal')
    plot_n(peak_dict[1], axeses[1], label='Apnea')
    axeses[1].set_ylabel('Number')
    axeses[1].set_xlabel('peak prominences')

    for axes in axeses:
        axes.legend()
    if args.show:
        pyplot.show()
    if args.figure_path:
        fig.tight_layout()
        fig.savefig(args.figure_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
