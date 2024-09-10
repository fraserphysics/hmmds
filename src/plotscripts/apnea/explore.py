""" explore.py Makes a figure with hr, low pass, band pass, respiration

Here is an example of use:

python explore.py --heart_rate_path_format build/derived_data/ECG/{}_self_AR3/heart_rate \
  --root ../../.. --model_sample_frequency 4 explore.pdf

The result is sort of like a view from the GUI apnea/explore.py
"""

import sys
import argparse

import pint
import numpy

import plotscripts.utilities
import hmmds.applications.apnea.utilities

PINT = pint.get_application_registry()  # Makes objects from pickle.load


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Make one of the figures illustrating apnea data')
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument(
        '--show',
        action='store_true',
        help='display figure in pop-up window rather than storing it as a pdf')

    parser.add_argument('fig_path',
                        type=str,
                        help='Path for storing the result, eg, explore.pdf')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def plot_signal(signal: numpy.ndarray,
                sample_frequency,
                key: str,
                axes,
                t_start,
                t_stop,
                marker=None):
    """Create x values and plot subsequence of signal

    Args:
        signal: Y values
        sample_frequency: Pint scalar with dimension 1/t
        key: For plot legend
        axes: A matplotlib axes object
        t_start: Beginning of segment to plot
        t_stop: End of segment to plot
    """

    times, n_start, n_stop = interval2times(t_start, t_stop, sample_frequency)

    if marker:
        axes.plot(times.to('minutes').magnitude,
                  signal[n_start:n_stop],
                  marker=marker,
                  color='black',
                  linestyle='',
                  markersize=8)
    else:
        axes.plot(times.to('minutes').magnitude,
                  signal[n_start:n_stop],
                  label=key)
    axes.legend()
    return


def depint(time, frequency=100 * PINT('Hz')):
    """Return index for time sampled at frequency

    """
    return int((time * frequency).to('').magnitude)


def interval2times(start, stop, frequency=100 * PINT('Hz')):
    """Maps a time interval and frequency to an array of times

    Args:
        start: Pint time
        stop: Pint time

    Returns: times

    """
    n_start = depint(start, frequency)
    n_stop = depint(stop, frequency) + 1
    assert n_stop - n_start > 5
    return numpy.arange(n_start, n_stop) / frequency, n_start, n_stop


def main(argv=None):
    """Make first 4 plots of apnea chapter.  They are all time series.

    """

    args, matplotlib, pyplot = plotscripts.utilities.import_and_parse(
        parse_args, argv)

    fig, (ax_heart_rate, ax_low_pass, ax_band_pass,
          ax_respiration) = pyplot.subplots(nrows=4, ncols=1, sharex=True)

    heart_rate = hmmds.applications.apnea.utilities.HeartRate(args, 'a03')
    heart_rate.filter_hr()

    assert heart_rate.hr_sample_frequency.to('Hz').magnitude == 2
    assert heart_rate.model_sample_frequency.to('1/minute').magnitude == 4

    t_start, t_stop = (x * PINT('minute') for x in (423, 429))

    for signal, key, axes in (
        (heart_rate.raw_hr, 'a03 Raw Heart Rate', ax_heart_rate),
        (heart_rate.slow, 'Low Pass', ax_low_pass),
        (heart_rate.resp_pass, 'Band Pass', ax_band_pass),
        (heart_rate.envelope, 'Envelope', ax_band_pass),
            #
        (heart_rate.respiration, 'Respiration', ax_respiration)):

        axes.yaxis.set_major_locator(pyplot.MaxNLocator(3))
        plot_signal(signal, heart_rate.hr_sample_frequency, key, axes, t_start,
                    t_stop)

    plot_signal(heart_rate.get_slow(),
                heart_rate.model_sample_frequency,
                '',
                ax_low_pass,
                t_start,
                t_stop,
                marker='.')

    plot_signal(heart_rate.get_respiration(),
                heart_rate.model_sample_frequency,
                '',
                ax_respiration,
                t_start,
                t_stop,
                marker='.')

    if not args.show:
        fig.savefig(args.fig_path)
    else:
        pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
