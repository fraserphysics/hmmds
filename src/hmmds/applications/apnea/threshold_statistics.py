"""threshold_statistics.py Explore dependence of threshold on statistics

python threshold_statistics.py c_model default threshold_statistics.pdf a01 a02 ...

Derived from shift_threshold.py
"""
from __future__ import annotations

import sys
import argparse
import typing
import pickle
import os.path

import numpy
import scipy.optimize

import utilities
import plotscripts.utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Plot to check if statistics predict best threshold")
    utilities.common_arguments(parser)
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('c_model_path', type=str, help="path to model")
    parser.add_argument('a_model_path', type=str, help="path to model")
    parser.add_argument('fig_path', type=str, help="path to result")
    parser.add_argument('record_names', type=str, nargs='+')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def minimum_error(model_record):
    """Find rough approximation of threshold that minimizes error for
    a record.

    """
    thresholds = numpy.geomspace(.11, 20000, 10)
    objective_values = numpy.zeros(len(thresholds), dtype=int)
    counts = numpy.empty((len(thresholds), 4), dtype=int)
    for i, threshold in enumerate(thresholds):
        model_record.classify(threshold=threshold)
        counts[i, :] = model_record.score()
        objective_values[i] = counts[i, 1] + counts[i, 2]
    best_i = numpy.argmin(objective_values)
    return thresholds[best_i], counts[best_i]


class Statistics:

    def __init__(self, model_records, args):
        """Calaulate characteristics of training data for setting thresholds

        Args:
            model_records: EG, model_records['a01'] is a ModelRecord instance
            args: Command line arguments
        """
        self.args = args
        self.pass1 = {}
        self.z = {}
        self.log_threshold = {}

        sum_lt = 0
        sum_lt_sq = 0
        sum_lt_psd = 0
        sum_psd = 0
        sum_psd_sq = 0
        for model_record in model_records.values():
            pass1 = utilities.Pass1(model_record.record_name, args)
            n_a = model_record.class_from_expert.sum()
            log_t = numpy.log10(find_threshold(model_record, n_a)[0])
            self.log_threshold[model_record.record_name] = log_t
            psd = pass1.psd

            sum_lt += log_t
            sum_lt_sq += log_t * log_t
            sum_lt_psd += log_t * psd
            sum_psd += psd
            sum_psd_sq += psd * psd

        n_records = len(model_records)
        self.mean_lt = sum_lt / n_records
        self.dev_lt = numpy.sqrt(sum_lt_sq / n_records -
                                 self.mean_lt * self.mean_lt)
        mean_lt_psd = sum_lt_psd / n_records
        self.mean_psd = sum_psd / n_records
        self.dev_psd = numpy.sqrt(sum_psd_sq / n_records -
                                  self.mean_psd * self.mean_psd)
        self.correlation = (mean_lt_psd - self.mean_lt * self.mean_psd) / (
            self.dev_psd * self.dev_lt)

    def analyze(self: Statistics, model_record: utilities.ModelRecord,
                low: float, high: float):
        """Map PSD to: Linear forecast; Sum over range; z vector

        Args:
            model_record:
            low: Lower limit for sum in cycles per minute
            high: Upper limit in cpm

        """
        # pass1.psd ranges from 1e2 to 1e-4
        # pass1.frequencies ranges from 0 to 60
        # pass1.bandpower(a, b) sums psd in frequencies from a to b in cpm
        # 2,000 channels so resolution is 60/2,000 = 0.03 cpm
        pass1 = utilities.Pass1(model_record.record_name, self.args)
        z = (pass1.psd - self.mean_psd) / self.dev_psd
        z_correlation = self.correlation * z

        # argmax finds first place inequality is true
        channel_low = max(0, numpy.argmax(pass1.frequencies > low))
        channel_high = numpy.argmax(pass1.frequencies > high)
        if channel_high <= 0:
            channel_high = len(pass1.frequencies)

        result = z_correlation[channel_low:channel_high].sum()
        #print(f'{model_record.record_name=} {result=}')
        return result, z[channel_low:channel_high].sum(), z


def main(argv=None):
    """Plot best thresholds on training data against various statistics

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    model_records = dict((name, utilities.ModelRecord(args.a_model_path, name))
                         for name in args.record_names)

    count_sum = 0
    for name, model_record in model_records.items():
        threshold, count = minimum_error(model_record)
        count_sum += count
    fraction = (count_sum[1] + count_sum[2]) / count_sum.sum()
    print(f'{count_sum} {count_sum[1]+count_sum[2]} {fraction}')
    return 0

    statistics = Statistics(model_records, args)  # 11 seconds

    errors = errors_abcd(a, b, c, d, model_records, statistics)
    print(f'{errors=}')

    fig, (min_axes, a_axes, b_axes, c_axes,
          d_axes) = pyplot.subplots(nrows=5, figsize=(6, 12))

    # Plot the function f_abcd
    z_s = numpy.linspace(-5800, 2500, 100)
    y = numpy.array(list(statistics.f_abcd(a, b, c, d, z=z) for z in z_s))
    min_axes.semilogy(z_s, y)

    for name, model_record in model_records.items():
        f_psd, z_sum, z = statistics.analyze(model_record, 0, 60)
        min_threshold = minimum_error(model_record)
        min_axes.semilogy(
            -z_sum,
            min_threshold,
            marker=f'${name}$',
            markersize=14,
            linestyle='None',
        )
        min_axes.set_xlabel('-z_sum')
        min_axes.set_ylabel('Min Threshold')

    errors = list(
        errors_abcd(a_, b, c, d, model_records, statistics) for a_ in a_s)
    a_axes.plot(a_s, errors)
    a_axes.set_xlabel('a')
    a_axes.set_ylabel('error count')

    errors = list(
        errors_abcd(a, b_, c, d, model_records, statistics) for b_ in b_s)
    b_axes.plot(b_s, errors)
    b_axes.set_xlabel('b')
    b_axes.set_ylabel('error count')

    errors = list(
        errors_abcd(a, b, c_, d, model_records, statistics) for c_ in c_s)
    c_axes.semilogx(c_s, errors)
    c_axes.set_xlabel('c')
    c_axes.set_ylabel('error count')

    errors = list(
        errors_abcd(a, b, c, d_, model_records, statistics) for d_ in d_s)
    d_axes.semilogx(d_s, errors)
    d_axes.set_xlabel('d')
    d_axes.set_ylabel('error count')

    # Put tight_layout after plots
    fig.tight_layout()
    fig.savefig(args.fig_path)

    if args.show:
        pyplot.show()

    return 0


if __name__ == "__main__":
    if sys.argv[0] == "write_shift_statistics.py":
        sys.exit(write_shift_statistics())
    sys.exit(main())
