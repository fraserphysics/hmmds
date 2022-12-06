"""spectrogram.py: Plot a specrogram of a heart rate time series

python spectrogram.py heart_rate_dir annotations_file record_name output_file

"""

import sys
import argparse
import pickle
import os

import pint

import numpy
import matplotlib

import respire, utilities, rtimes2hr

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as pyplot

PINT = pint.UnitRegistry()
pint.set_application_registry(PINT)  # Makes objects from pickle.load
# use this pint registry.


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Plot spectrogram')
    parser.add_argument('--HR_dir',
                        type=str,
                        default='../../../../build/derived_data/apnea/Lphr/',
                        help='Path to heart rate data')
    parser.add_argument('--rtimes_dir',
                        type=str,
                        default='../../../../build/derived_data/apnea/Rtimes/',
                        help='Path to heart rate data')
    parser.add_argument(
        '--annotations',
        type=str,
        default='../../../../raw_data/apnea/summary_of_training')
    parser.add_argument('--name', type=str, default='a11')
    parser.add_argument('--time_window',
                        type=float,
                        nargs=2,
                        help='Restrict plot to times in this window in minutes')
    parser.add_argument('--frequency_window',
                        type=float,
                        nargs=2,
                        help='Restrict plot to frequencies in this window')
    parser.add_argument('--sample_rate_in',
                        type=int,
                        default=2,
                        help='Samples per second of input')
    parser.add_argument('--sample_rate_out',
                        type=int,
                        default=10,
                        help='Samples per minute for results')
    parser.add_argument('--fft_width',
                        type=int,
                        default=128,
                        help='Number of samples for each fft')
    parser.add_argument(
        '--deviation_w',
        type=int,
        default=4,
        help='width of context for calculating heart beat deviations')
    parser.add_argument('output', type=str, help='Path to result')
    args = parser.parse_args(argv)
    args.sample_rate_in *= PINT('Hz')
    args.sample_rate_out /= PINT('minutes')
    return args


def main(argv=None):
    """Make spectrogram figure
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    # Read heart rate data
    with open(os.path.join(args.HR_dir, args.name + '.lphr'), 'rb') as _file:
        hr_dict = pickle.load(_file)
    time_series = hr_dict['hr_band_pass']
    hr_dt = 1 / hr_dict['sample_frequency'].to('1/minute')

    rtimes = rtimes2hr.read_rtimes(
        os.path.join(args.rtimes_dir, args.name + '.rtimes')).magnitude
    hr_deviations = utilities.rtimes2dev(rtimes, args.deviation_w) * PINT('Hz')

    # Calculate spectrogram with unit lengths
    frequencies, spec_times, psds = respire.spectrogram(hr_deviations, args)
    assert psds.shape == (len(frequencies), len(spec_times))
    norms = numpy.sqrt((psds * psds).sum(axis=0))
    psds /= norms

    annotations = utilities.read_expert(args.annotations, args.name)

    fig, (ax_time_series, ax_spectrogram, ax_annotation) = pyplot.subplots(3, 1)

    ax_annotation.get_shared_x_axes().join(ax_time_series, ax_spectrogram,
                                           ax_annotation)
    if args.time_window:
        t_start, t_stop = (t * PINT('minutes') for t in args.time_window)
        n_start, n_stop = (int((t / hr_dt).to('')) for t in (t_start, t_stop))
    else:
        n_start = 0
        n_stop = len(time_series) - 1
    ax_time_series.plot(
        numpy.arange(n_start, n_stop) * hr_dt.to('minutes').magnitude,
        time_series[n_start:n_stop].to('1/minutes').magnitude)
    ax_time_series.set_xticklabels([])
    ax_time_series.set_ylabel('band pass heart rate/cpm')

    if args.time_window:
        n_start, n_stop = numpy.searchsorted(
            spec_times.to('minutes').magnitude, args.time_window)
    else:
        n_start = 0
        n_stop = len(spec_times) - 1

    if args.frequency_window:
        f_start, f_stop = numpy.searchsorted(
            frequencies.to('1/minute').magnitude, args.frequency_window)
    else:
        f_start = 0
        f_stop = len(frequencies) - 1
    times_minutes = spec_times.to('minutes').magnitude[n_start:n_stop]
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
    ax_spectrogram.set_xticklabels([])

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

    pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
