"""statistic_plots.py.  Make plots for choosing statistics for pass1.

python statistic_plots.py  --show foo.pdf

Plot 1. Plot the PSD of the unfiltered heart rate.

"""

import sys
import argparse
import pickle

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
    parser.add_argument('--X_names',
                        type=str,
                        nargs='+',
                        default=[f'x{x:02d}' for x in range(1, 36)])
    parser.add_argument('--B_names',
                        type=str,
                        nargs='+',
                        default=[f'b{x:02d}' for x in range(1, 5)])
    parser.add_argument('--sample_rate_in',
                        type=int,
                        default=2,
                        help='Samples per second of input')
    parser.add_argument('--fft_width',
                        type=int,
                        default=4096,
                        help='Number of samples for each fft')
    parser.add_argument('--data_dir',
                        type=str,
                        default='../../../../build/derived_data/ECG/',
                        help='Path to heart rate data for reading')
    parser.add_argument(
        '--model',
        type=str,
        default='../../../../build/derived_data/apnea/models/two_ar3_masked6.1',
        help='Path to model')
    parser.add_argument('--format',
                        type=str,
                        default='{0}/{1}_self_AR3/heart_rate',
                        help='Map from (data_dir,name) to file of heart_rates')
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

    def __init__(self, name, args, normalization=0.29):
        """Set the following attributes:

        name: EG "a01"

        heart_rate: Time series in cycles per minute

        psd: Power spectral density estimate / sum of frequencies above normalization

        cdf: Cumulative distribution of low pass heart rate estimate

        percents: Values at a few cdf points

        """
        best_power = 1.757  # Raise likelihood of interval to this power

        self.likelihood = utilities.interval_hmm_likelihood(
            args.model, name, best_power)

        self.name = name
        self.y_data = hmm.base.JointSegment(
            utilities.read_slow_respiration(args, name))
        self.cdf = self.y_data['slow'].copy()
        self.cdf.sort()
        self.cdf /= self.cdf[int(0.5 * len(self.cdf))]
        self.percents = dict(
            (frac, self.cdf[int(frac * len(self.cdf))])
            for frac in (0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.995))
        with open(args.format.format(args.data_dir, name), 'rb') as _file:
            _dict = pickle.load(_file)
            sample_frequency = _dict['sample_frequency'].to(
                '1/minutes').magnitude
            hr = _dict['hr']
            trim = int(sample_frequency *
                       args.trim_start.to('minutes').magnitude)
        self.heart_rate = _dict['hr'].to('1/minute').magnitude[trim:]
        self.frequencies, raw_psd = scipy.signal.welch(self.heart_rate,
                                                       fs=sample_frequency,
                                                       nperseg=args.fft_width)
        channel = numpy.argmax(self.frequencies > normalization)
        self.psd = raw_psd  / raw_psd[channel:].sum()

    def statistic_3(self):
        return self.likelihood

    def statistic_2(self):
        """Linear function of CDF of slow heart rate estimate.

        Try <D,CDF> where D is the difference between the combined a
        records and the combined c records?

        """
        a = numpy.array([-1.0, 1.0, -1.0])
        v = numpy.array([self.percents[x] for x in (.01, .8, .995)])
        return numpy.dot(a, v)

    def statistic_1(self, low=1.29, high=3.63):
        """Spectral power in the range of low frequency apnea
        oscillations.  Range in cpm

        """
        channel_low = numpy.argmax(self.frequencies > low)
        channel_high = numpy.argmax(self.frequencies > high)
        return self.psd[channel_low:channel_high].sum()


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]
    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    trouble = 'a10 c10'.split()
    colors = ['r', 'b']
    fig, axeses = pyplot.subplots(nrows=3, figsize=(6, 8))
    fig.tight_layout()

    records = dict(
        (name, Record(name, args))
        for name in args.A_names + args.C_names + args.X_names + args.B_names)
    frequencies = records[args.A_names[0]].frequencies

    def plot_psd(names, label, color):
        psd_list = [records[name].psd for name in names]
        for name in names:
            assert numpy.array_equal(frequencies, records[name].frequencies)
        psd_average = sum(psd_list) / len(psd_list)
        for axes in axeses[:2]:
            axes.plot(frequencies,
                      numpy.log(psd_average),
                      label=label,
                      color=color,
                      linewidth=2,
                      linestyle='solid')
        #for psd in psd_list:
        #    axeses[0].plot(frequencies, numpy.log(psd), color=color, linestyle='dotted')

    plot_psd(args.A_names, r'$\bar a$', 'r')
    plot_psd(args.C_names, r'$\bar c$', 'b')
    for axes in axeses[:2]:
        for name, color in zip(trouble, colors):
            axes.plot(frequencies,
                      numpy.log(records[name].psd),
                      color=color,
                      label=name)
        axes.set_xlabel('frequency')
        axes.set_ylabel('log psd')

    axeses[1].set_xlim(40, 125)
    axeses[1].set_ylim(-7, -4)

    def plot_combined_cdf(names, color):
        combined = numpy.concatenate([records[name].cdf for name in names])
        combined.sort()
        y = numpy.linspace(0, 1, len(combined))
        axeses[1].plot(combined,
                       y,
                       color=color,
                       linewidth=2,
                       linestyle='dashed')

    def plot_cdf(name, color=None, linestyle='solid', label=False):
        x = records[name].cdf
        y = numpy.linspace(0, 1, len(x))
        if color is None:
            axeses[1].plot(x, y, label=name, linestyle=linestyle)
            return
        if not label:
            axeses[1].plot(x, y, color=color, linestyle=linestyle)
            return
        axeses[1].plot(x, y, color=color, linestyle=linestyle, label=name)

    # plot_combined_cdf(args.A_names, 'r')
    # plot_combined_cdf(args.C_names, 'b')
    # for name in trouble:
    #     plot_cdf(name, label=True)
    # axeses[1].set_xlabel('Heart Rate / median')
    # axeses[1].set_ylabel('CDF Low Pass HR')

    def plot_stats(names, color):
        for name in names:
            axeses[2].plot(records[name].statistic_1(),
                           records[name].statistic_3(),
                           color=color,
                           marker=f'${name}$',
                           markersize=14,
                           linestyle='None')

    y = numpy.linspace(1.0e4, -2.0e5, 2)
    x = .228 - y * 0.045 / 1.0e5
    axeses[2].plot(x, y)
    plot_stats(args.A_names, 'r')
    plot_stats(args.C_names, 'b')
    plot_stats(args.B_names, 'g')
    plot_stats(args.X_names, 'k')
    axeses[2].set_xlabel(r'$F(PSD)$')
    axeses[2].set_ylabel(r'$G(CDF)$')
    for axes in axeses[:2]:
        axes.legend()
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
