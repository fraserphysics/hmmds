"""plot_pp.py Characterize distribution of peaks in the heart rate signal

python plot_pp.py --records a01 a02 --heart_rate_sample_frequency 24 --show pp_plot.pdf

Make a scatter plot of prominence and period (time between peaks)

"""
import sys
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
    utilities.common_arguments(parser)
    parser.add_argument('figure_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def analyze_records(args):
    """Calculate (prominence, period) pairs.  Also find boundaries for
    digitizing prominence.

    """

    f_sample = args.heart_rate_sample_frequency.to('1/minute').magnitude
    apnea_key = 1

    # Calculate (prominence, period) pairs
    peak_dict = {0: [], 1: []}
    for record_name in args.a_names:  # FixMe: Could be args.records
        raw_dict = utilities.read_slow_class(args, record_name)
        slow = raw_dict['slow']
        _class = raw_dict['class']
        peaks, properties = utilities.peaks(slow,
                                            args.heart_rate_sample_frequency)
        for index in range(len(peaks) - 1):
            t_peak = peaks[index]
            prominence_t = properties['prominences'][index]
            period_t = (peaks[index + 1] - t_peak) / f_sample
            class_t = _class[t_peak]
            peak_dict[class_t].append((prominence_t, period_t))

    # Calculate boundaries for prominence based on peaks during apnea
    pp_array = numpy.array(peak_dict[apnea_key]).T
    apnea_peaks = pp_array[0]
    apnea_peaks.sort()
    boundaries = []
    for index in range(0, len(apnea_peaks), 1300):
        boundaries.append(apnea_peaks[index])
    boundaries = numpy.array(boundaries).T

    return peak_dict, boundaries


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


def plot_histograms(axes, peak_dict, boundaries, bin=3):
    """Plot x=interval y=frequency

    Args:
        axes: Matplotlib axes
        peak_dict: (prominence, interval) = peak_dict[0][i] for normal
        boundaries: For prominence from apnea data
        bin: Plot histogram of intervals for data with prominence in this bin.  Min = 1 Max = 6

    """
    n_key, a_key = (0, 1)
    prom_key, interval_key = (0, 1)

    # Get intervals for all normal and apnea data
    intervals = [
        numpy.array(peak_dict[key])[:, interval_key] for key in (n_key, a_key)
    ]

    # Get intervals for data with prominence in bin
    a_prominence, a_interval = (numpy.array(peak_dict[a_key])[:, key]
                                for key in (prom_key, interval_key))
    locations = numpy.nonzero(
        numpy.digitize(a_prominence, boundaries) == bin)[0]
    intervals.append(a_interval[locations])
    #values = [peak_dict[key][:,1][numpy.nonzero(peak_dict[key][:,0]-1)] for key in (0,1)]

    axes.hist(intervals,
              bins=numpy.linspace(.4, 3, 20),
              label=f'N A A_{bin}'.split(),
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
    peak_dict, boundaries = analyze_records(args)

    plot_scatter(ax_scatter, peak_dict)
    plot_histograms(ax_histogram, peak_dict, boundaries)
    if args.show:
        pyplot.show()
    fig.savefig(args.figure_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
