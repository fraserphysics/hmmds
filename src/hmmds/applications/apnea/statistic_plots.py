"""statistic_plots.py.  Make plots for choosing statistics for pass1.

Each plot has one trace for the C files and another trace for the A
files.  The first plot is the cumulative distribution of the
normalized respiration signal, and the second plot is a PSD estimate.

Plot 1. Call the respiration signal r[t].  For the ith record,
calculate M_i the median value of r and define R[t] = r[t]/M_i.  Then
plot the cumulative distribution of all of the R[t] values for all of
the records.

Plot 2. Plot the PSD of the unfiltered heart rate using a window size
of 1024 seconds.  Channel 25 will correspond to apnea osscilations of
40.96 seconds.

"""
# ToDo: Label frequency axis in cpm

import sys
import os.path
import argparse
import pickle
import typing

import pint
import numpy
import numpy.linalg
import scipy.signal

import utilities
import plotscripts.utilities

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Plots for choosing statistics')
    parser.add_argument('--A_names', type=str, nargs='+',
                        default=[f'a{x:02d}' for x in range(1,21)])
    parser.add_argument('--C_names', type=str, nargs='+',
                        default=[f'c{x:02d}' for x in range(1,11)])
    parser.add_argument('--sample_rate_in',
                        type=int,
                        default=2,
                        help='Samples per second of input')
    parser.add_argument('--fft_width',
                        type=int,
                        default=2048,
                        help='Number of samples for each fft')
    parser.add_argument('--data_dir',
                        type=str,
                        default='../../../../build/derived_data/ECG/',
                        help='Path to heart rate data for reading')
    parser.add_argument(
        '--format', type=str, default='{0}/{1}_self_AR3/heart_rate',
        help='Map from (data_dir,name) to file of heart_rates')
    parser.add_argument('--fig_dir',
                        type=str,
                        default='.',
                        help='Path for writing figures')
    parser.add_argument('--show',
                        action='store_false',
                        help="display figure using Qt5")
    args = parser.parse_args(argv)
    args.sample_rate_in *= PINT('Hz')
    return args

class Record:
    def __init__(self, name, args):
        self.name = name
        path = args.format.format(args.data_dir, name)
        with open(path, 'rb') as _file:
            pickle_dict = pickle.load(_file)
        if name == 'c04':
            self.heart_rate = pickle_dict['hr'].to('1/minute').magnitude[2520:]
        else:
            self.heart_rate = pickle_dict['hr'].to('1/minute').magnitude
        frequencies, self.psd = scipy.signal.welch(self.heart_rate, nperseg=args.fft_width)
        
        raw = utilities.filter_hr(
            self.heart_rate,
            0.5 * PINT('seconds'),
            low_pass_width=2 * numpy.pi / (15 * PINT('seconds')),
            bandpass_center=2 * numpy.pi * 14 / PINT('minutes'),
            skip=1)['respiration']
        raw.sort()
        self.respiration = raw/(raw[int(.5 * len(raw))])
        

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    fig, (psd_axes, resp_axes, stats_axes) = pyplot.subplots(nrows=3, figsize=(6,8))

    records = {}
    for name in args.A_names + args.C_names:
        records[name] = Record(name, args)

    def plot_psd(names, label, color):
        psd_list = [records[name].psd for name in names]
        psd_average = sum(psd_list)/len(psd_list)
        psd_axes.plot(numpy.log(psd_average), label=label, color=color, linewidth=2)
        for psd in psd_list:
            psd_axes.plot(numpy.log(psd), color=color, linestyle='dotted')
        psd_axes.plot(numpy.log(records['c04'].psd), color='k', label='c04')
    plot_psd(args.A_names, 'a', 'r')
    plot_psd(args.C_names, 'c', 'b')
    psd_axes.legend()
    psd_axes.set_xlim(10,70)
    psd_axes.set_ylim(3.5, 9.5)

    def plot_respiration(names, label, color):
        resp_list = [records[name].respiration for name in names]
        combined = numpy.concatenate(resp_list)
        combined.sort()
        y = numpy.linspace(0,1, len(combined))
        resp_axes.plot(combined, y, label=label, color=color, linewidth=2)
        for name in names:
            respiration = records[name].respiration
            y = numpy.linspace(0,1,len(respiration))
            if name == 'c04':
                resp_axes.plot(respiration, y, color='k', label='c04')
            else:
                resp_axes.plot(respiration, y, color=color, linestyle='dotted', linewidth=1)

    plot_respiration(args.A_names, 'a', 'r')
    plot_respiration(args.C_names, 'c', 'b')
    resp_axes.legend()
    resp_axes.plot((0.64, 0.64), (0, .5))
    resp_axes.set_xlim(0,1)

    def plot_stats(names, color):
        for name in names:
            stat_1 = records[name].psd[12:55].sum()
            resp_ = records[name].respiration
            stat_2 = numpy.searchsorted(resp_, 0.64)/len(resp_)
            stats_axes.plot(stat_1, stat_2, color=color, marker=f'${name}$', markersize=14, linestyle = 'None')
    plot_stats(args.A_names, 'r')
    plot_stats(args.C_names, 'b')
    pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
