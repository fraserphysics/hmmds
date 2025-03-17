""" all_ecgs.py Plot all ecgs six per figure.

Grid imitates an ECG Strip, see https://www.rnceus.com/ekg/ekghowto.html

"""
import sys
import argparse
import pickle
import os
import math

import numpy
import pint

import plotscripts.utilities

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Plot ECG and decoded states')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--time_interval',
                        type=float,
                        nargs=2,
                        default=[210, 210.1])
    parser.add_argument('--y_range', type=float, nargs=2, default=[-2.5, 3])
    parser.add_argument('ecg_dir', type=str, help='Path to data')
    parser.add_argument('fig_dir', type=str, help="directory for results")
    return parser.parse_args(argv)


def get_samples(a_in, space):
    '''Sample the interval spanned by a_in at integer spacings of
    space

    Args:
        a_in: Sorted sequence of numbers
        space: Distance between samples

    '''
    bottom = space * math.ceil(a_in[0] / space)
    top = space * round(a_in[-1] / space)
    result = numpy.linspace(bottom, top, round((top-bottom)/space)+1)
    return result


def main(argv=None):
    """Make 6 time ecg series pictures

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    t_start, t_stop = (t * PINT('minutes') for t in args.time_interval)
    y_min, y_max = args.y_range

    # Set tick locations
    y_minor_ticks = numpy.arange(y_min, y_max + 0.05, 0.1)
    y_major_ticks = get_samples(y_minor_ticks, 0.5)
    y_label_values = get_samples(y_minor_ticks, 2.0)
    y_label_indices = numpy.array(
        numpy.searchsorted(y_major_ticks, y_label_values))
    y_labels_text = [f'{y:.0f}' for y in y_label_values]
    y_labels = [''] * len(y_major_ticks)
    for label, index in zip(y_labels_text, y_label_indices):
        y_labels[index] = label

    s_start, s_stop = (t.to('seconds').magnitude for t in (t_start, t_stop))
    x_minor_ticks = numpy.arange(s_start, s_stop + .01, 0.04)
    x_major_ticks = get_samples(x_minor_ticks, .2)
    x_labels = [''] * len(x_major_ticks)

    x_label_values = get_samples(x_major_ticks, 1.0)
    x_label_indices = numpy.array(
        numpy.searchsorted(x_major_ticks, x_label_values))
    formatted = [
        plotscripts.utilities.format_time(t * PINT('seconds'))
        for t in x_label_values
    ]
    for label, index in zip(formatted, x_label_indices):
        x_labels[index] = label

    def plot_six(records):

        fig, axeses_3x2 = pyplot.subplots(nrows=3,
                                          ncols=2,
                                          sharex=True,
                                          sharey=True,
                                          figsize=(7, 9))
        for axes in axeses_3x2[2, :]:
            axes.set_xlabel('time H:M:S')
        for axes in axeses_3x2[:, 0]:
            axes.set_ylabel('ECG/mV')
        for axes, name in zip(axeses_3x2.flatten(), records):
            with open(os.path.join(args.ecg_dir, f'{name}'), 'rb') as _file:
                _dict = pickle.load(_file)
            ecg = _dict['ecg']
            ecg_times = _dict['times'] * PINT('seconds')
            n_start, n_stop = numpy.searchsorted(
                ecg_times.to('seconds').magnitude,
                [t.to('seconds').magnitude for t in (t_start, t_stop)])
            times = ecg_times[n_start:n_stop].to('seconds').magnitude
            axes.plot(times, ecg[n_start:n_stop], label=name)
            axes.set_ylim(y_min, y_max)
            axes.set_xlim(s_start, s_stop)
            axes.legend()
            # From Google AI "matplotlib EKG grid"
            # Customize the grid
            axes.grid(which='major',
                      linestyle='-',
                      linewidth='0.5',
                      color='red')
            axes.grid(which='minor',
                      linestyle='-',
                      linewidth='0.5',
                      alpha=.1,
                      color='red')

            axes.set_xticks(x_major_ticks, x_labels)
            axes.set_xticks(x_minor_ticks, minor=True)
            axes.set_yticks(y_major_ticks, y_labels)
            axes.set_yticks(y_minor_ticks, minor=True)

        fig.tight_layout()
        return fig

    record_names = os.listdir(args.ecg_dir)
    record_names.remove('flag')
    record_names.sort()
    fig_dict = {}
    for i in range(0, len(record_names) + 6, 6):
        if i >= len(record_names):
            break
        last = min(i + 6, len(record_names))
        fig_dict[f'ecgs_{record_names[i]}'] = plot_six(record_names[i:last])

    if args.show:
        pyplot.show()
        return 0
    for name, fig in fig_dict.items():
        fig_path = os.path.join(args.fig_dir, name)
        fig.savefig(fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
