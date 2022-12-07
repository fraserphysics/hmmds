""" apnea_ts_plots.py makes the first four figures for Chapter 6

Here is an example of use:

python apnea_ts_plots.py --data_dir derived_data/apnea  figs/apnea/a03erA.pdf

"""
import sys
import argparse
import os.path
import pickle

import pint
import numpy
import matplotlib  # type: ignore

PINT = pint.UnitRegistry()

# For PhysioNet data
samples_per_second = 100 * PINT('Hz')


def depint(time, frequency=100*PINT('Hz')):
    """Return index for time sampled at frequency

    """
    return int((time*frequency).to('').magnitude)


def interval2times(start, stop, frequency=100*PINT('Hz')):
    """Maps a time interval and frequency to an array of times

    Args:
        start: Pint time
        stop: Pint time

    Returns: times

    """
    return numpy.arange(depint(start, frequency), depint(stop, frequency))/frequency


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


def read_data(data_file: str) -> dict:  # FixMe: do this in line
    """Read in "data_file" as a dict

    This is only used to read a03er.
    """
    with open(data_file, 'rb') as _file:
        _dict = pickle.load(_file)
    return _dict


def read_lphr(data_file: str) -> numpy.ndarray:  # FixMe: do this in line
    """Read in "data_file" as an array"""
    with open(data_file, 'rb') as _file:
        hr_dict = pickle.load(_file)
    print(f"""
    hr_dict.keys={hr_dict.keys()}
    hr_dict['sample_frequency']={hr_dict['sample_frequency']}
    hr_dict['hr_low_pass'][0]={hr_dict['hr_low_pass'][0]}
    """)
    
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
    with open(os.path.join(args.data_dir, 'a03er.pickle'), 'rb') as _file:
        a03er_dict = pickle.load(_file)
    times = numpy.arange(len(a03er_dict['ECG'])) / samples_per_second

    def subplot(ax,
                y_data,
                times,
                y_label,
                label_times,
                label_yvalues=None,
                ylim=None,
                yscale=1):
        """Put a trace on an axes

        Args:
            ax: Matplotlib axes
            y_data: Y values to plot
            times: Array of integers for the x-axis
            y_label: Label for y axis
            label_times: Pint times for labels on x axis
            label_yvalues: Place tick marks on y axis
            ylim: Range of y axis
            yscale: Multiply y_data by this factor
        """
        n_start = times[0]
        n_stop = times[-1] + 1
        ax.plot(times, y_data[n_start:n_stop], 'k-')
        ax.set_ylabel(y_label)
        ax.set_xticks([depint(time) for time in label_times])
        ax.set_xticklabels([format_time(t) for t in label_times])
        if label_yvalues is not None:
            ax.set_yticks(numpy.array(label_yvalues) * yscale)
            ax.set_yticklabels([r'$% 2.0f$' % x for x in label_yvalues])
        ax.set_ylim(ylim)
        ax.set_xlim((n_start, n_stop))

    _times = interval2times(start, stop).to('centiseconds').magnitude.astype(int)
    subplot(ax_ecg, a03er_dict['ECG'], _times, r'$ECG$', label_times, numpy.arange(-1,4))
    subplot(ax_onr, a03er_dict['Resp N'], _times, r'$ONR$', label_times, [-.5, 0, .5])
    subplot(ax_O2, a03er_dict['SpO2'], _times, r'$SpO_2$', label_times, numpy.arange(6,10)*10)

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
    label_times = [t*PINT('minutes') for t in (70, 71, 72)]
    fig = plot_ECG_ONR_O2(args, time_start, time_stop, label_times,
                          subplots)
    return fig


@register
def a03HR(args, subplots):
    """For third figure in Chapter 6: "a03HR.pdf"

    Subplots of heart rate and O2 saturation
    """
    a03_lphr, sample_frequency = read_lphr(os.path.join(args.data_dir, 'Lphr/a03.lphr'))
    a03er_dict = read_data(os.path.join(args.data_dir, 'a03er.pickle'))
    t_start, t_stop = (t*PINT('minutes') for t in (55, 65))
    n_times = interval2times(t_start, t_stop).to('centiseconds').magnitude.astype(int)
    n_start = n_times[0]
    n_stop = n_times[-1] + 1

    ylim = (40, 90)  # For lphr only

    fig, (ax_hr, ax_O2) = subplots(nrows=2, ncols=1, sharex=True)

    ax_hr.plot(n_times, a03_lphr[n_start:n_stop], 'k-')
    ax_hr.set_ylabel(r'$HR$')
    label_yvalues = numpy.arange(45, 86, 10)
    ax_hr.set_yticks(label_yvalues)
    ax_hr.set_yticklabels(['$% 2.0f$' % x for x in label_yvalues])
    ax_hr.set_ylim(ylim)
    ax_hr.set_xlim(n_start, n_stop)

    ax_O2.plot(n_times, a03er_dict['SpO2'][n_start:n_stop], 'k-')
    ax_O2.set_ylabel(r'$SpO_2$')
    x_label_times = numpy.arange(55, 66, 5)*PINT('minutes')
    ax.set_xticks([depint(time) for time in x_label_times])
    ax.set_xticklabels([format_time(t) for t in x_label_times])
    label_yvalues = numpy.arange(60, 101, 10)
    ax_O2.set_yticks(label_yvalues)
    ax_O2.set_yticklabels(['$% 2.0f$' % x for x in label_yvalues])

    return fig


@register
def ApneaNLD(args, subplots):
    """ Two segments of Heart rate for the fourth figure in Chapter 6.

    Subplots of heart rate for segments of a01 and a12
    """
    a01_lphr = read_lphr(os.path.join(args.data_dir, 'Lphr/a01.lphr'))
    a12_lphr = read_lphr(os.path.join(args.data_dir, 'Lphr/a12.lphr'))
    fig, (ax_a01, ax_a12) = subplots(nrows=2, ncols=1, sharex=False)

    ax_a01.plot(a01_lphr, 'k-')  # FixMe: Want data for x-axis
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

    ax_a12.plot(a12_lphr, 'k-')  # FixMe: Want data for x-axis
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
                        default='../../../build/derived_data/apnea',
                        help='Directory that contains Lphr')
    parser.add_argument(
        '--show',
        action='store_true',
        help='display figure in pop-up window rather than storing it as a pdf')

    parser.add_argument(
        'fig_path',
        type=str,
        help='Path for storing the result, eg, ./../figs/apnea/a03HR.pdf')
    args = parser.parse_args(argv)

    import matplotlib  # FixMe: Should not be necessary
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
