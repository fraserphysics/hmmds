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
# ToDo: Label frequency axis in cpm.  Use pint to track pickled sample rates.

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
import hmm.base

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Plots for choosing statistics')
    parser.add_argument('--A_names',
                        type=str,
                        nargs='+',
                        default=[f'a{x:02d}' for x in range(1, 21)])
    parser.add_argument('--C_names',
                        type=str,
                        nargs='+',
                        default=[f'c{x:02d}' for x in range(1, 11)])
    parser.add_argument('--a_models',
                        type=str,
                        nargs='+',
                        default='a06 a09 a10 a11'.split(),
                        help='models to use for statistic_3')
    parser.add_argument('--c_models',
                        type=str,
                        nargs='+',
                        default='c02 c07 c09 c10'.split(),
                        help='models to use for statistic_3')
    parser.add_argument('--model_template',
                        type=str,
                        default='%s/%s_unmasked',
                        help='For paths to models')
    parser.add_argument('--model_dir',
                        type=str,
                        default='../../../../build/derived_data/apnea/models',
                        help='Path to trained models')
    parser.add_argument('--X_names',
                        type=str,
                        nargs='+',
                        default=[f'x{x:02d}' for x in range(1, 36)])
    parser.add_argument('--sample_rate_in',
                        type=int,
                        default=2,
                        help='Samples per second of input')
    parser.add_argument('--threshold',
                        type=float,
                        default=0.46,
                        help='For respiration statistic')
    parser.add_argument('--fft_width',
                        type=int,
                        default=2048,
                        help='Number of samples for each fft')
    parser.add_argument('--data_dir',
                        type=str,
                        default='../../../../build/derived_data/ECG/',
                        help='Path to heart rate data for reading')
    parser.add_argument('--format',
                        type=str,
                        default='{0}/{1}_self_AR3/heart_rate',
                        help='Map from (data_dir,name) to file of heart_rates')
    parser.add_argument('--fig_dir',
                        type=str,
                        default='.',
                        help='Path for writing figures')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('fig_path', type=str, help="path to figure")
    utilities.common_arguments(parser)
    args = parser.parse_args(argv)
    args.sample_rate_in *= PINT('Hz')
    utilities.join_common(args)
    return args


class Record:

    def __init__(self, name, args):
        """Set the following attributs:

        name: EG "a01"

        heart_rate: Time series in cycles per minute

        psd: Power spectral density estimate / sum of all but first 11
             channels

        respiration: Minimum/maximum of the respiration signal over
                     each minute.  The values are sorted.  My idea is
                     that during apnea the resipration signal has
                     large variations.

        y_data:      Data for calculating likelihood of hmm

        """
        self.name = name
        self.y_data = hmm.base.JointSegment(
            utilities.read_slow_respiration(
                args, name))
        path = args.format.format(args.data_dir, name)
        with open(path, 'rb') as _file:
            pickle_dict = pickle.load(_file)
        self.heart_rate = pickle_dict['hr'].to('1/minute').magnitude[2424:]
        frequencies, raw_psd = scipy.signal.welch(self.heart_rate,
                                                  nperseg=args.fft_width)
        self.psd = raw_psd / raw_psd[11:].sum()

        raw = utilities.filter_hr(
            self.heart_rate,
            0.5 * PINT('seconds'),
            low_pass_width=2 * numpy.pi / (15 * PINT('seconds')),
            bandpass_center=2 * numpy.pi * 14 / PINT('minutes'),
            skip=1)['respiration']
        len_minutes = len(raw) // 120
        self.respiration = numpy.empty(len_minutes)
        for minute in range(len_minutes):
            interval = slice(minute * 120, minute * 120 + 120)
            min_ = raw[interval].min()
            max_ = raw[interval].max()
            if max_ > 0:
                self.respiration[minute] = min_ / max_
            else:
                self.respiration[minute] = 1.0
        self.respiration.sort()
    def statistic_1(self):
        """Spectral power in the range of low frequency apnea
        oscillations

        """
        return self.psd[22:62].sum()
    def statistic_2(self, threshold):
        """The fraction of minutes in which the min/max is below
        threshold.

        """
        return numpy.searchsorted(self.respiration, threshold)/len(self.respiration)
    def statistic_3(self, a_models, c_models):
        """ The difference between the log likelihoods of a_models and c_models for self.y_data.

        """
        def total(models):
            result = 0
            for name, model in models.items():
                likelihood = model.likelihood(self.y_data)
                if likelihood.min() > 0:
                    result += numpy.log(likelihood).sum() / len(self.y_data)
                else:
                    print(f'likelihood[{name}]({self.name}) = 0')
                    result -= 2
            return result
        return total(a_models) - total(c_models)

def read_models(names, args):
    result = {}
    for name in names:
        model_path = args.model_template % (args.model_dir, name)
        with open(model_path, 'rb') as _file:
            old_args, result[name] = pickle.load(_file)
    return result
        
def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    trouble = 'c02 c07 c09 c10 a06 a09 a10'.split()
    fig, (psd_axes, respiration_axes, stats_axes) = pyplot.subplots(nrows=3,
                                                                    figsize=(6,
                                                                             8))

    records = {}
    for name in args.A_names + args.C_names + args.X_names:
        records[name] = Record(name, args)
    a_models = read_models(args.a_models, args)
    c_models = read_models(args.c_models, args)

    def plot_psd(names, label, color):
        psd_list = [records[name].psd for name in names]
        psd_average = sum(psd_list) / len(psd_list)
        psd_axes.plot(numpy.log(psd_average),
                      label=label,
                      color=color,
                      linewidth=4)
        for psd in psd_list:
            psd_axes.plot(numpy.log(psd), color=color, linestyle='dotted')

    plot_psd(args.A_names, 'a', 'r')
    plot_psd(args.C_names, 'c', 'b')
    for name in trouble:
        psd_axes.plot(numpy.log(records[name].psd), label=name)
    psd_axes.legend()

    #psd_axes.set_xlim(0,200)
    #psd_axes.set_ylim(3.5, 9.5)

    def plot_respiration(names, label, color):
        respiration_list = [records[name].respiration for name in names]
        combined = numpy.concatenate(respiration_list)
        combined.sort()
        y = numpy.linspace(0, 1, len(combined))
        respiration_axes.plot(combined,
                              y,
                              label=label,
                              color=color,
                              linewidth=2)
        for name in names:
            if name in trouble:
                respiration = records[name].respiration
                y = numpy.linspace(0, 1, len(respiration))
                if name[0] == 'c':
                    respiration_axes.plot(respiration, y, label=name)
                else:
                    respiration_axes.plot(respiration,
                                      y,
                                      label=name,
                                      linestyle='dotted')

    plot_respiration(args.A_names, 'a', 'r')
    plot_respiration(args.C_names, 'c', 'b')
    respiration_axes.legend()
    respiration_axes.plot((args.threshold, args.threshold), (0, 1))

    def plot_stats(names, color):
        for name in names:
            stats_axes.plot(
                records[name].statistic_1(),
                records[name].statistic_2(args.threshold),
                #records[name].statistic_3(a_models, c_models),
                color=color,
                marker=f'${name}$',
                markersize=14,
                linestyle='None')

    plot_stats(args.A_names, 'r')
    plot_stats(args.C_names, 'b')
    plot_stats(args.X_names, 'k')
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
