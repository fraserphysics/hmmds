"""spectrogram.py: Plot a specrogram of a heart rate time series

python spectrogram.py heart_rate_file output_file

"""

import sys
import argparse
import pickle

import pint

import numpy
import matplotlib

import respire, utilities

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as pyplot

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Plot spectrogram')
    parser.add_argument(
        '--annotations',
        type=str,
        default='../../../../raw_data/apnea/summary_of_training')
    parser.add_argument('--record', type=str, default='a11')
    parser.add_argument('--time_window',
                        type=float,
                        nargs=2,
                        help='Restrict plot to times in this window in minutes')
    parser.add_argument(
        '--frequency_window',
        type=float,
        nargs=2,
        default=(-1, -1),  #
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
                        default=64,
                        help='Number of samples for each fft')
    parser.add_argument('HR_file', type=str, help='Path to heart rate data')
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

    with open(args.HR_file, 'rb') as _file:
        hr_dict = pickle.load(_file)
    time_series = hr_dict['hr_band_pass']
    # Ugliness to get hr_dt in pint registry for this module
    hr_dt = (1 / hr_dict['sample_frequency'].to('1/minute')
            ).magnitude * PINT('minutes')

    annotations = utilities.read_expert(args.annotations, args.record)
    frequencies, spec_times, psds = respire.spectrogram(time_series, args)
    assert psds.shape == (len(frequencies), len(spec_times))

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
        time_series[n_start:n_stop])

    if args.time_window:
        n_start, n_stop = numpy.searchsorted(
            spec_times.to('minutes').magnitude, args.time_window)
    else:
        n_start = 0
        n_stop = len(spec_times) - 1
    times_minutes = spec_times.to('minutes').magnitude[n_start:n_stop]
    ax_spectrogram.pcolormesh(times_minutes,
                              frequencies.to('1/minute').magnitude,
                              -10 * numpy.log10(psds[:, n_start:n_stop]),
                              cmap=matplotlib.cm.hsv)
    ax_spectrogram.set_xlabel('t/minutes')
    ax_spectrogram.set_ylabel('f/cpm')

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
    # figure, axes = pyplot.subplots(1, 1, figsize=(6, 2))
    # axes.plot(times, hr, label='hr')
    # axes.plot(times, hr_low_pass, label='hr_low')
    # axes.plot(times, hr_band_pass, label='hr_band')
    # axes.legend(loc='upper right')

    pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
