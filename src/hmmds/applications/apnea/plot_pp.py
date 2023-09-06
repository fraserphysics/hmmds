"""plot_pp.py Characterize distribution of peaks in the heart rate signal

python plot_pp.py --records a01 a02 --heart_rate_sample_frequency 24 --show pp_plot.pdf

Make a scatter plot of prominence and period (time between peaks)

"""
# An earlier version of this code used scipy.stats.gamma to estimate
# pdfs of the intervals between peaks of the heart rate for normal and
# apnea minutes.  Because fits were not satisfactory, I wrote
# density_ratio.py which uses Gaussian kernels to estimate the ratio
# of the pdfs.
import sys
import argparse
import pickle

import numpy

import utilities
import plotscripts.utilities
import apnea_ratio


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Analyze peaks of heart rate sign")
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    utilities.common_arguments(parser)
    parser.add_argument('figure_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def plot_record(args, record_name, axes):
    """ For debugging reading
    """
    raw_dict = utilities.read_slow_class(args, record_name)
    slow = raw_dict['slow'] / 10
    times = numpy.arange(
        len(slow)) / args.heart_rate_sample_frequency.to('1/minute').magnitude
    axes.plot(times, slow, label='slow')
    axes.plot(times, raw_dict['class'], label='class')


def plot_scatter(axes, peak_dict):
    """Plot x=prominence y=interval

    """
    for key, color, _class in ((0, 'b', 'N'), (1, 'r', 'A')):
        x_y = numpy.array(peak_dict[key])
        axes.scatter(x_y[:, 0],
                     x_y[:, 1],
                     label=_class,
                     color=color,
                     marker='.',
                     s=5)
    axes.set_xlabel('Prominence')
    axes.set_ylabel('Interval')
    axes.legend()


def plot_histograms(axes, peak_dict, boundaries, plot_bin=3):
    """Plot x=interval y=frequency

    Args:
        axes: Matplotlib axes
        peak_dict: (prominence, interval) = peak_dict[0][i] for normal
        boundaries: For prominence from apnea data
        plot_bin: Plot histogram of intervals for data with prominence in this bin.  Min = 1 Max = 6

    """
    n_key, a_key = (0, 1)
    prom_key, interval_key = (0, 1)
    intervals = []

    # Get intervals for data with prominence in each bin
    a_prominence, a_interval = (numpy.array(peak_dict[a_key])[:, key]
                                for key in (prom_key, interval_key))
    digits = numpy.digitize(a_prominence, boundaries)
    for _bin in range(len(boundaries) + 1):
        locations = numpy.nonzero(digits == _bin)[0]
        intervals.append(a_interval[locations])

    # Get intervals for all normal and apnea data
    for key in (a_key, n_key):
        intervals.append(numpy.array(peak_dict[key])[:, interval_key])

    axes.hist(intervals[-2:] + [intervals[plot_bin]],
              bins=numpy.linspace(.4, 6, 40),
              label=f'N A A_{plot_bin}'.split(),
              log=False,
              density=True)

    axes.legend()


def main(argv=None):
    """ Read specified records and make plot
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    fig, (ax_scatter, ax_histogram) = pyplot.subplots(nrows=2, figsize=(6, 8))

    # Find peaks
    peak_dict, boundaries = apnea_ratio.analyze_records(args, args.a_names)

    plot_scatter(ax_scatter, peak_dict)
    plot_histograms(ax_histogram, peak_dict, boundaries)
    if args.show:
        pyplot.show()
    fig.savefig(args.figure_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
