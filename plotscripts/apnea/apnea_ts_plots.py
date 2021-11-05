""" apnea_ts_plots.py makes the first four figures for Chapter 6

Here is an example of use:

python apnea_ts_plots.py --data_dir derived_data/apnea  figs/apnea/a03erA.pdf

This is the first line of the data file a03er_seg and the fields:

3000.01 -0.65 -4814.0 -4002.0  -56.0     70.0
seconds ECG                    1000*ONR  O2sat

This is the first line of the data file a03.lphr and the fields:

0.0   70.5882352941 2.88250231173
minutes lphr        bandpass hr

"""
import sys
import argparse
import os.path

import numpy
import matplotlib


def minutes_seconds(x: int) -> str:
    """Format seconds as minutes:seconds for LaTeX

    Args:
        x: Integer number of seconds since start of record

    Returns:
        string formatted for LaTeX
    """
    return r'$%1d\!:\!%02d$' % (x / 60, x % 60)


def read_data(data_file):
    """Read in "data_file" as an array"""
    with open(data_file, 'r') as _file:
        data = [[float(x) for x in line.split()] for line in _file.readlines()]
    return numpy.array(data).T


def plot_ECG_ONR_O2(args, start: int, stop: int, time_interval: tuple,
                    label_times: tuple, subplots):
    """Plot Electrocardiogram, oronasal airflow, and O2 Saturation

    Args:
        args: Command line arguments (for data directory)
        start: Start plots at this line in the file a03er_seg
        stop:  Stop plots at this line in the file a03er_seg
        time_interval: Time range for the plots in seconds
        label_times: Put time labels at these times
        subplots: matplotlib.pyplot function
"""
    fig, (ax_ecg, ax_onr, ax_O2) = subplots(nrows=3, ncols=1, sharex=True)
    a03er_segment = read_data(os.path.join(args.data_dir, 'a03er_seg'))

    def subplot(ax,
                y_data,
                y_label,
                label_times,
                ylim,
                label_yvalues,
                yscale=1):
        """Put a trace on an axes

        Args:
            ax: Matplotlib axes
            y_data: Y values to plot
            y_label: Label for y axis
            label_times: Labels for time axis
            ylim: Range of y axis
            label_yvalues: Place tick marks on y axis
            yscale: Multiply y_data by this factor
        """
        seconds_per_minute = 60
        ax.plot(a03er_segment[0, start:stop] / seconds_per_minute, y_data, 'k-')
        ax.set_ylabel(y_label)
        ax.set_xticks(label_times)
        ax.set_xticklabels([minutes_seconds(t) for t in label_times])
        ax.set_yticks(numpy.array(label_yvalues) * yscale)
        ax.set_yticklabels([r'$% 2.0f$' % x for x in label_yvalues])
        ax.set_ylim(ylim)
        ax.set_xlim(time_interval)

    subplot(ax_ecg, a03er_segment[1, start:stop] / 1000, r'$ECG$', [],
            (-0.015, 0.035), numpy.arange(-10, 31, 10), 1e-3)
    subplot(ax_onr, a03er_segment[4, start:stop] / 1000, r'$ONR$', [],
            (-15, 15), numpy.arange(-10, 11, 10))
    subplot(ax_O2, a03er_segment[5, start:stop], r'$SpO_2$', label_times,
            (55, 100), numpy.arange(60, 101, 15))

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
    line_start = 45000  # about 3450 seconds or 57.5 minutes
    line_stop = 57005  # about 3570 seconds
    time_range = (57.5, 59.5)
    time_labels = (58, 59)
    fig = plot_ECG_ONR_O2(args, line_start, line_stop, time_range, time_labels,
                          subplots)
    return fig


@register
def a03erN(args, subplots):
    """For second figure in Chapter 6: "a03erN.pdf"

    Plots of ECG, ONR and SpO2
    """
    line_start = 120000  # About 4200 seconds
    line_stop = 132001  # About 4320 seconds
    time_range = (70, 72)
    time_labels = (70, 71, 72)
    fig = plot_ECG_ONR_O2(args, line_start, line_stop, time_range, time_labels,
                          subplots)
    return fig


@register
def a03HR(args, subplots):
    """For third figure in Chapter 6: "a03HR.pdf"

    Subplots of heart rate and O2 saturation
    """
    a03_lphr = read_data(os.path.join(args.data_dir, 'low_pass_heart_rate/a03'))
    a03er_segment = read_data(os.path.join(args.data_dir, 'a03er_seg'))

    xlim = (55, 65)  # For both
    ylim = (40, 90)  # For lphr only

    fig, (ax_hr, ax_O2) = subplots(nrows=2, ncols=1, sharex=True)

    ax_hr.plot(a03_lphr[0, :], a03_lphr[1, :], 'k-')
    ax_hr.set_ylabel(r'$HR$')
    label_yvalues = numpy.arange(45, 86, 10)
    ax_hr.set_yticks(label_yvalues)
    ax_hr.set_yticklabels(['$% 2.0f$' % x for x in label_yvalues])
    ax_hr.set_xticks([])
    ax_hr.set_ylim(ylim)
    ax_hr.set_xlim(xlim)

    ax_O2.plot(a03er_segment[0, :] / 60, a03er_segment[5, :], 'k-')
    ax_O2.set_ylabel(r'$SpO_2$')
    label_times = numpy.arange(55, 66, 5)
    ax_O2.set_xticks(label_times)
    ax_O2.set_xticklabels([minutes_seconds(x) for x in label_times])
    label_yvalues = numpy.arange(60, 101, 10)
    ax_O2.set_yticks(label_yvalues)
    ax_O2.set_yticklabels(['$% 2.0f$' % x for x in label_yvalues])
    ax_O2.set_xlim(xlim)

    return fig


@register
def ApneaNLD(args, subplots):
    """ Two segments of Heart rate for the fourth figure in Chapter 6.

    Subplots of heart rate for segments of a01 and a12
    """
    a01_lphr = read_data(os.path.join(args.data_dir, 'low_pass_heart_rate/a01'))
    a12_lphr = read_data(os.path.join(args.data_dir, 'low_pass_heart_rate/a12'))
    fig, (ax_a01, ax_a12) = subplots(nrows=2, ncols=1, sharex=False)

    ax_a01.plot(a01_lphr[0, :], a01_lphr[1, :], 'k-')
    ax_a01.set_ylabel(r'$a01$HR')
    t_label_values = numpy.arange(115, 126, 5)
    ax_a01.set_xticks(t_label_values)
    ax_a01.set_xticklabels([minutes_seconds(x) for x in t_label_values])
    y_label_values = numpy.arange(40, 101, 20)
    ax_a01.set_yticks(y_label_values)
    ax_a01.set_yticklabels(['$% 3.0f$' % x for x in y_label_values])
    # Crop the plot
    ax_a01.set_ylim(40, 100)
    ax_a01.set_xlim(115, 125)

    ax_a12.plot(a12_lphr[0, :], a12_lphr[1, :], 'k-')
    ax_a12.set_ylabel(r'$a12$HR')
    t_label_values = numpy.arange(570, 577, 5)
    ax_a12.set_xticks(t_label_values)
    ax_a12.set_xticklabels([minutes_seconds(x) for x in t_label_values])
    y_label_values = numpy.arange(40, 81, 20)
    ax_a12.set_yticks(y_label_values)
    ax_a12.set_yticklabels(['$% 3.0f$' % x for x in y_label_values])
    # Crop the plot
    ax_a12.set_ylim(40, 80)
    ax_a12.set_xlim(568, 577)

    return fig


def main(argv=None):
    """Make first 4 plots of apnea chapter.  They are all time series.

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='Make one of the figures illustrating apnea data')
    parser.add_argument('--data_dir',
                        type=str,
                        default='../../derived_data/apnea',
                        help='Directory that contains low_pass_heart_rate')
    parser.add_argument(
        '--show',
        action='store_true',
        help='display figure in pop-up window rather than storing it as a pdf')

    parser.add_argument(
        'fig_path',
        type=str,
        help='Path for storing the result, eg, ./../figs/apnea/a03HR.pdf')
    args = parser.parse_args(argv)

    if args.show:
        matplotlib.use('Qt5Agg')
    else:
        matplotlib.use('PDF')  # Permits absence of enviroment variable DISPLAY
    import matplotlib.pyplot  # pylint: disable=import-outside-toplevel,redefined-outer-name

    params = {
        'axes.labelsize': 18,  # Plotting parameters for latex
        'font.size': 15,
        'legend.fontsize': 15,
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Computer Modern Roman',
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'figure.autolayout': True
    }
    matplotlib.rcParams.update(params)

    key = os.path.splitext(os.path.basename(args.fig_path))[0]

    if key in PLOTS:
        fig = PLOTS[key](args, matplotlib.pyplot.subplots)
        if not args.show:
            fig.savefig(args.fig_path)
        else:
            matplotlib.pyplot.show()
    else:
        raise ValueError("""Don't know how to make figure {0}.
args:{1}""".format(key, args))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
