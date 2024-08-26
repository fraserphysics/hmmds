"""spectrogram.py: Make a figure with a spectrogram of a heart signal

"""

from __future__ import annotations  # Enables, eg, (self: HMM,

import sys
import argparse
import pickle

import pint

import numpy
import matplotlib

from hmmds.applications.apnea import utilities

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as pyplot

PINT = pint.get_application_registry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Plot spectrogram')
    utilities.common_arguments(parser)
    parser.add_argument('--record_name', type=str, default='a11')
    parser.add_argument('--time_window',
                        type=float,
                        nargs=2,
                        help='Restrict plot to times in this window in minutes')
    parser.add_argument('--frequency_window',
                        type=float,
                        nargs=2,
                        help='Restrict plot to frequencies in this window')
    parser.add_argument('--show',
                        action='store_true',
                        help='display figure in pop-up window')
    parser.add_argument('input', type=str, help='Path to spectrogram data')
    parser.add_argument('annotations',
                        type=str,
                        help='Path to expert annotations')
    parser.add_argument('output', type=str, help='Path to result')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def main(argv=None):
    """Make spectrogram figure
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    # Read spectrogram data
    with open(args.input, 'rb') as _file:
        sgram_dict = pickle.load(_file)
    frequencies = sgram_dict['frequencies']
    times = sgram_dict['times']
    psds = sgram_dict['psds']
    time_series = sgram_dict['time_series']
    hr_dt = sgram_dict['hr_dt']

    # Calculate spectrogram with unit lengths
    assert psds.shape == (len(frequencies), len(times))
    norms = numpy.sqrt((psds * psds).sum(axis=0))
    psds /= norms

    # Read expert annotations
    annotations = utilities.read_expert(args.annotations, args.record_name)

    figure, (ax_time_series, ax_spectrogram,
             ax_annotation) = pyplot.subplots(3, 1, sharex=True)

    # Plot time series
    if args.time_window:
        t_start, t_stop = (t * PINT('minutes') for t in args.time_window)
        n_start, n_stop = (int((t / hr_dt).to('')) for t in (t_start, t_stop))
    else:
        n_start = 0
        n_stop = len(time_series) - 1
    ax_time_series.plot(
        numpy.arange(n_start, n_stop) * hr_dt.to('minutes').magnitude,
        time_series[n_start:n_stop].to('1/minutes').magnitude)
    ax_time_series.set_ylim([55, 95])
    ax_time_series.set_yticks([60, 80])
    ax_time_series.set_ylabel('HR/cpm')

    # Plot spectrogram
    if args.time_window:
        n_start, n_stop = numpy.searchsorted(
            times.to('minutes').magnitude, args.time_window)
    else:
        n_start = 0
        n_stop = len(times) - 1
    if args.frequency_window:
        f_start, f_stop = numpy.searchsorted(
            frequencies.to('1/minute').magnitude, args.frequency_window)
    else:
        f_start = 0
        f_stop = len(frequencies) - 1
    times_minutes = times.to('minutes').magnitude[n_start:n_stop]
    frequencies_cpm = frequencies.to('1/minute').magnitude[f_start:f_stop]
    z = -10 * numpy.log10(psds[f_start:f_stop, n_start:n_stop])
    ax_spectrogram.pcolormesh(
        times_minutes,
        frequencies_cpm,
        z,
        cmap=matplotlib.cm.hsv,
        shading='gouraud',
    )
    ax_spectrogram.set_ylabel('f/cpm')
    ax_spectrogram.set_yticks([10, 20])

    # Plot annotations
    def annotation(time):
        """annotation(15.7) -> 0 or 1, 0 for normal

        Args:
            time: In minutes
        """
        i_time = int(time)  # int(.9) = 0
        if i_time >= len(annotations):
            return annotations[-1]
        return annotations[i_time]

    ax_annotation.plot(times_minutes,
                       list(annotation(time) for time in times_minutes))
    ax_annotation.set_xlabel('t/minutes')
    ax_annotation.set_ylim(-0.2, 1.2)
    ax_annotation.set_yticks([0, 1])
    ax_annotation.set_yticklabels(['$N$', '$A$'])

    if args.time_window:
        ax_annotation.set_xlim(*args.time_window)
    figure.savefig(args.output)
    if args.show:
        pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
