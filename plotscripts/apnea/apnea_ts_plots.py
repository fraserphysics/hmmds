""" apnea_ts_plots.py a03er_seg a03_lphr a01_lphr a12_lphr a03erA_plot
         a03erN_plot a03HR_plot ApneaNLD

This is the first line of the data file a03er_seg and the fields:

3000.01 -0.65 -4814.0 -4002.0  -56.0     70.0
time     ECG                   1000*ONR  O2sat

This is the first line of the data file a03.lphr and the fields:

0.0   70.5882352941 2.88250231173
time  lphr          bandpass hr
"""
import sys
import argparse
import os.path

import numpy as np

minutes_seconds = lambda x: '$%1d\!:\!%02d$' % (x / 60, x % 60
                                               )  # Latex time format


def read_data(data_file):
    """Read in "data_file" as an array"""
    f = open(data_file, 'r')
    data = [[float(x) for x in line.split()] for line in f.readlines()]
    f.close()
    return np.array(data).T


def plot_ECG_ONR_O2(args, start: int, stop: int, xbounds: tuple,
                    label_times: tuple, subplots):
    """Plot Electrocardiogram, oronasal airflow, and O2 Saturation

    Args:
        args: Command line arguments (for data directory)
        start: Start plots at this line in the file a03er_seg
        stop:  Stop plots at this line in the file a03er_seg
        xbounds: Time range for the plots in seconds
        label_times: Put time labels at these times
        subplots: matplotlib.pyplot function
"""
    fig, (axecg, axonr, axO2) = subplots(nrows=3, ncols=1, sharex=True)
    seg = read_data(os.path.join(args.data_dir, 'a03er_seg'))

    def subplot(ax,
                y_data,
                y_label,
                label_times,
                ylim,
                label_yvalues,
                yscale=1):
        ax.plot(seg[0, start:stop] / 60, y_data, 'k-')
        ax.set_ylabel(y_label)
        ax.set_xticks(label_times)
        ax.set_xticklabels([minutes_seconds(x) for x in label_times])
        ax.set_yticks(np.array(label_yvalues) * yscale)
        ax.set_yticklabels([r'$% 2.0f$' % x for x in label_yvalues])
        ax.set_ylim(ylim)
        ax.set_xlim(xbounds)
        return

    subplot(axecg, seg[1, start:stop] / 1000, r'$ECG$', [], (-0.015, 0.035),
            np.arange(-10, 31, 10), 1e-3)
    subplot(axonr, seg[4, start:stop] / 1000, r'$ONR$', [], (-15, 15),
            np.arange(-10, 11, 10))
    subplot(axO2, seg[5, start:stop], r'$SpO_2$', label_times, (55, 100),
            np.arange(60, 101, 15))

    return fig


PLOTS = {}  # keys are function names, values are functions


def register(func):
    """Decorator that puts function in PLOTS dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    PLOTS[func.__name__] = func
    return func


@register
def a03erA(args, subplots):
    """For first figure in Chapter 6: "a03erA_plot.pdf"

    Plots of ECG, ONR and SpO2
    """
    line_start = 45000
    line_stop = 57005
    time_range = (57.5, 59.5)
    time_labels = (58, 59)
    fig = plot_ECG_ONR_O2(args, line_start, line_stop, time_range, time_labels,
                          subplots)
    return fig


@register
def a03erN(args, subplots):
    """For second figure in Chapter 6: "a03erN_plot.pdf"

    Plots of ECG, ONR and SpO2
    """
    line_start = 120000
    line_stop = 132001
    time_range = (70, 72)
    time_labels = (70, 71, 72)
    fig = plot_ECG_ONR_O2(args, line_start, line_stop, time_range, time_labels,
                          subplots)
    return fig


@register
def a03HR(args, subplots):
    """For third figure in Chapter 6: "a03HR_plot.pdf"

    """
    a03_lphr = read_data(os.path.join(args.data_dir, 'low_pass_heart_rate/a03'))
    seg = read_data(os.path.join(args.data_dir, 'a03er_seg'))

    xlim = (55, 65)  # For both
    ylim = (40, 90)  # For lphr only

    fig, (axhr, axO2) = subplots(nrows=2, ncols=1, sharex=True)

    axhr.plot(a03_lphr[0, :], a03_lphr[1, :], 'k-')
    axhr.set_ylabel(r'$HR$')
    label_yvalues = np.arange(45, 86, 10)
    axhr.set_yticks(label_yvalues)
    axhr.set_yticklabels(['$% 2.0f$' % x for x in label_yvalues])
    axhr.set_xticks([])
    axhr.set_ylim(ylim)
    axhr.set_xlim(xlim)

    axO2.plot(seg[0, :] / 60, seg[5, :], 'k-')
    axO2.set_ylabel(r'$SpO_2$')
    label_times = np.arange(55, 66, 5)
    axO2.set_xticks(label_times)
    axO2.set_xticklabels([minutes_seconds(x) for x in label_times])
    label_yvalues = np.arange(60, 101, 10)
    axO2.set_yticks(label_yvalues)
    axO2.set_yticklabels(['$% 2.0f$' % x for x in label_yvalues])
    axO2.set_xlim(xlim)

    return fig


@register
def ApneaNLD(args, subplots):
    """ Two segments of Heart rate for the fourth figure in Chapter 6.

    Need a01_lphr, a12_lphr
    """
    a01_lphr = read_data(os.path.join(args.data_dir, 'low_pass_heart_rate/a01'))
    a12_lphr = read_data(os.path.join(args.data_dir, 'low_pass_heart_rate/a12'))
    fig, (ax_a01, ax_a12) = subplots(nrows=2, ncols=1, sharex=False)

    ax_a01.plot(a01_lphr[0, :], a01_lphr[1, :], 'k-')
    ax_a01.set_ylabel(r'$a01$HR')
    xrng = np.arange(115, 126, 5)
    ax_a01.set_xticks(xrng)
    ax_a01.set_xticklabels([minutes_seconds(x) for x in xrng])
    yrng = np.arange(40, 101, 20)
    ax_a01.set_yticks(yrng)
    ax_a01.set_yticklabels(['$% 3.0f$' % x for x in yrng])
    ax_a01.set_ylim(40, 100)
    ax_a01.set_xlim(115, 125)

    ax_a12.plot(a12_lphr[0, :], a12_lphr[1, :], 'k-')
    ax_a12.set_ylabel(r'$a12$HR')
    xrng = np.arange(570, 577, 5)
    ax_a12.set_xticks(xrng)
    ax_a12.set_xticklabels([minutes_seconds(x) for x in xrng])
    yrng = np.arange(40, 81, 20)
    ax_a12.set_yticks(yrng)
    ax_a12.set_yticklabels(['$% 3.0f$' % x for x in yrng])
    ax_a12.set_ylim(40, 80)
    ax_a12.set_xlim(568, 577)

    return fig


def main(argv=None):
    """

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='Make a figure illustrating apnea data')
    parser.add_argument('--data_dir',
                        type=str,
                        default='../../derived_data/apnea',
                        help='Directory that contains low_pass_heart_rate')
    parser.add_argument(
        '--show',
        action='store_true',
        help='display figure in pop-up window rather than storing it as a pdf')

    parser.add_argument('fig_path',
                        type=str,
                        help='Attach "pdf" to this name for figure file')
    args = parser.parse_args(argv)

    import matplotlib
    if args.show:
        matplotlib.use('Qt5Agg')
    else:
        matplotlib.use('PDF')  # Permits absence of enviroment variable DISPLAY
    import matplotlib.pyplot  # Must be after matplotlib.use

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
