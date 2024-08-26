""" apnea_ts_plots.py makes the first four figures for Chapter 6

Here is an example of use:

python apnea_ts_plots.py --heart_rate_path_format derived_data/ECG/{}_self_AR3/heart_rate  figs/apnea/a03erA.pdf

"""
import sys
import argparse
import os.path
import pickle

import pint
import numpy

import plotscripts.utilities
import hmmds.applications.apnea.utilities

PINT = pint.get_application_registry()  # Makes objects from pickle.load

# For PhysioNet data
samples_per_second = 100 * PINT('Hz')


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

    parser.add_argument(
        'fig_path',
        type=str,
        help='Path for storing the result, eg, ./../figs/apnea/a03HR.pdf')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


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
    return numpy.arange(depint(start, frequency), depint(stop,
                                                         frequency)) / frequency


def format_time(time: pint.Quantity) -> str:
    """Return a LaTeX representation of a pint time

    """
    float_seconds = time.to('seconds').magnitude
    assert float_seconds >= 0.0
    int_seconds = int(float_seconds)
    fraction = float_seconds - int_seconds
    int_minutes = int_seconds // 60
    seconds = int_seconds - 60 * int_minutes
    hours = int_minutes // 60
    minutes = int_minutes - 60 * hours
    if fraction < 0.01:
        return f'{hours:2d}:{minutes:02d}:{seconds:02d}'
    return f'${hours:2d}:{minutes:02d}:{int_seconds+fraction:05.2f}$'


def read_lphr(record: str) -> numpy.ndarray:
    """Read raw heart rate and return result as an array"""
    with open(data_file, 'rb') as _file:
        hr_dict = pickle.load(_file)
    return hr_dict['hr_low_pass'], hr_dict['sample_frequency']


def plot_ECG_ONR_O2(args, start: pint.Quantity, stop: pint.Quantity,
                    label_times: tuple, subplots):
    """Plot Electrocardiogram, oronasal airflow, and O2 Saturation

    Args:
        args: Command line arguments (for data directory)
        start: Time in the file a03er_seg
        stop:  Time in the file a03er_seg
        label_times: Put labels at these times
        subplots: matplotlib.pyplot function
"""
    fig, (ax_ecg, ax_onr, ax_O2) = subplots(nrows=3, ncols=1, sharex=True)
    with open(os.path.join(args.derived_apnea_data, 'a03er.pickle'),
              'rb') as _file:
        a03er_dict = pickle.load(_file)

    def subplot(ax,
                y_data,
                times,
                y_label,
                label_times,
                label_yvalues=None,
                ylim=None):
        """Put a trace on an axes

        Args:
            ax: Matplotlib axes
            y_data: Y values to plot
            times: Array of integers for the x-axis
            y_label: Label for y axis
            label_times: Pint times for labels on x axis
            label_yvalues: Place tick marks on y axis
            ylim: Range of y axis
        """
        n_start = times[0]
        n_stop = times[-1] + 1
        ax.plot(times, y_data[n_start:n_stop], 'k-')
        ax.set_ylabel(y_label)
        ax.set_xticks([depint(time) for time in label_times])
        ax.set_xticklabels([format_time(t) for t in label_times])
        if label_yvalues is not None:
            ax.set_yticks(numpy.array(label_yvalues))
            ax.set_yticklabels([f'$ {x:2.0f}$' for x in label_yvalues])
        ax.set_ylim(ylim)
        ax.set_xlim((n_start, n_stop))

    times = interval2times(start, stop).to('centiseconds').magnitude.astype(int)
    subplot(ax_ecg, a03er_dict['ECG'], times, r'$ECG$', label_times,
            numpy.arange(-1, 4))
    subplot(ax_onr, a03er_dict['Resp N'], times, r'$ONR$', label_times)
    subplot(ax_O2, a03er_dict['SpO2'], times, r'$SpO_2$', label_times,
            numpy.arange(6, 10) * 10)

    return fig


PLOTS = {}  # keys are function names, values are functions


def register(func):
    """Decorator that puts function in PLOTS dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    PLOTS[func.__name__] = func
    return func


@register
def a03erA(args, subplots):
    """For first figure in Chapter 6: "a03erA.pdf"

    Plots of ECG, ONR and SpO2
    """
    time_start = 57.5 * PINT('minutes')
    time_stop = 59.5 * PINT('minutes')
    label_times = (58 * PINT('minutes'), 59 * PINT('minutes'))
    fig = plot_ECG_ONR_O2(args, time_start, time_stop, label_times, subplots)
    return fig


@register
def a03erN(args, subplots):
    """For second figure in Chapter 6: "a03erN.pdf"

    Plots of ECG, ONR and SpO2
    """
    time_start = 70 * PINT('minutes')
    time_stop = 72 * PINT('minutes')
    label_times = [t * PINT('minutes') for t in (70, 71, 72)]
    fig = plot_ECG_ONR_O2(args, time_start, time_stop, label_times, subplots)
    return fig


def plot_heart_rate(args,
                    axes,
                    record_name,
                    t_start,
                    t_stop,
                    y_ticks,
                    x_ticks,
                    x_labels=True):
    """Read heart rate data and plot it on axes

    Args:
        axes: A matplotlib thing
        path: To the data
        t_start: Beginning of segment to plot
        t_stop: End of segment to plot
        y_ticks: Values for tick marks and labels
        x_ticks: Values for tick marks and perhaps labels
        x_labels: Flag for generating labels at tick marks
    """
    heart_rate = hmmds.applications.apnea.utilities.HeartRate(args, record_name)
    heart_rate.filter_hr()
    signal = heart_rate.get_slow() * PINT('1/minute')
    sample_frequency = heart_rate.model_sample_frequency  # or hr_sample_frequency

    times = interval2times(t_start, t_stop, sample_frequency)
    n_start = int((times[0] * sample_frequency).to('').magnitude)
    n_stop = int((times[-1] * sample_frequency).to('').magnitude) + 1

    axes.plot(
        times.to('minutes').magnitude,
        signal[n_start:n_stop].to('1/minute').magnitude, 'k-')

    axes.set_yticks(y_ticks)
    axes.set_yticklabels([f'${x:2.0f}$' for x in y_ticks])

    axes.set_xticks([time.to('minutes').magnitude for time in x_ticks])
    if x_labels:
        axes.set_xticklabels([format_time(time) for time in x_ticks])
    else:
        axes.set_xticklabels([])


@register
def a03HR(args, subplots):
    """For third figure in Chapter 6: "a03HR.pdf"

    Subplots of heart rate and O2 saturation
    """

    t_start, t_stop = [time * PINT('minutes') for time in (55, 65)]

    # sharex suppresses labels on ticks of upper plot
    fig, (ax_hr, ax_o2) = subplots(nrows=2, ncols=1, sharex=True)
    x_label_times = [time * PINT('minutes') for time in (55, 60, 65)]

    # Plot the heart rate in the upper plot
    plot_heart_rate(args, ax_hr, 'a03', t_start, t_stop,
                    numpy.arange(45, 86, 10), x_label_times)

    ax_hr.set_ylim(40, 90)
    ax_hr.set_ylabel(r'$HR$')

    # Plot SpO2 in lower plot
    with open(os.path.join(args.derived_apnea_data, 'a03er.pickle'),
              'rb') as _file:
        a03er_dict = pickle.load(_file)

    sample_frequency = 100 * PINT('Hz')  # For PhysioNet data
    times = interval2times(t_start, t_stop, sample_frequency)
    n_start = int((times[0] * sample_frequency).to('').magnitude)
    n_stop = int((times[-1] * sample_frequency).to('').magnitude) + 1

    ax_o2.plot(
        times.to('minutes').magnitude, a03er_dict['SpO2'][n_start:n_stop], 'k-')

    ax_o2.set_ylabel(r'$SpO_2$')
    label_yvalues = numpy.arange(60, 101, 10)
    ax_o2.set_yticks(label_yvalues)
    ax_o2.set_yticklabels(['$% 2.0f$' % x for x in label_yvalues])

    ax_o2.set_xticks([time.to('minutes').magnitude for time in x_label_times])
    ax_o2.set_xticklabels([format_time(time) for time in x_label_times])
    ax_o2.set_xlim(
        t_start.to('minutes').magnitude,
        t_stop.to('minutes').magnitude)

    return fig


@register
def ApneaNLD(args, subplots):
    """ Two segments of Heart rate for the fourth figure in Chapter 6.

    Subplots of heart rate for segments of a01 and a12
    """

    fig, (ax_a01, ax_a12) = subplots(nrows=2, ncols=1, sharex=False)

    t_start, t_stop = [time * PINT('minutes') for time in (115, 125)]
    x_label_times = [time * PINT('minutes') for time in (115, 120, 125)]

    plot_heart_rate(args, ax_a01, 'a01', t_start, t_stop,
                    numpy.arange(40, 105, 10), x_label_times)

    ax_a01.set_ylim(40, 100)
    ax_a01.set_ylabel(r'$a01 HR$')
    ax_a01.set_xlim(
        t_start.to('minutes').magnitude,
        t_stop.to('minutes').magnitude)

    t_start, t_stop = [time * PINT('minutes') for time in (568, 576)]
    x_ticks = [time * PINT('minutes') for time in (570, 575)]
    y_ticks = numpy.arange(40, 85, 10)

    plot_heart_rate(args, ax_a12, 'a12', t_start, t_stop, y_ticks, x_ticks)

    ax_a12.set_ylim(40, 85)
    ax_a12.set_ylabel(r'$a12 HR$')
    ax_a12.set_xlim(
        t_start.to('minutes').magnitude,
        t_stop.to('minutes').magnitude)

    return fig


def main(argv=None):
    """Make first 4 plots of apnea chapter.  They are all time series.

    """

    args, matplotlib, pyplot = plotscripts.utilities.import_and_parse(
        parse_args, argv)

    # If fig_path is fig_dir/a03HR.pdf then key is a03HR
    key = os.path.splitext(os.path.basename(args.fig_path))[0]

    if key in PLOTS:
        fig = PLOTS[key](args, pyplot.subplots)
        if not args.show:
            fig.savefig(args.fig_path)
        else:
            pyplot.show()
    else:
        raise ValueError("""Don't know how to make figure {0}.
args:{1}""".format(key, args))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
