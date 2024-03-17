"""fit2threshold.py Plots for function from record to threshold.
Derived from shift_threshold.py

python fit2threshold.py $(MULTI_BEST) threshold_statistics.pkl --mb 0.94 0.006 --records $(APLUSNAMES) --show

"""
from __future__ import annotations

import sys
import argparse
import typing
import pickle

import numpy
import scipy.optimize

import utilities
import plotscripts.utilities
from threshold_statistics import Fit


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Plot best threshold and N_A against F(PSD), the pass1 statistic")
    utilities.common_arguments(parser)
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--fig_path',
                        type=str,
                        help="path to result",
                        default="shift_threshold.pdf")
    parser.add_argument('model_path', type=str, help="path to model")
    parser.add_argument('statistics_path', type=str, help="path to model")
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def errors_mb(m: float, b: float, model_records: utilities.ModelRecord,
              statistics: Statistics) -> int:
    """Report number of errors for thresholds function

    Args:
        m: Threshold = 10**(m*log_fit + b)
        b: see m
        model_records: Dict of ModelRecord instances
    """
    counts = numpy.zeros(4, dtype=int)
    for model_record in model_records.values():
        threshold = statistics.f_mb_name(m, b, model_record.record_name)
        model_record.classify(threshold=threshold)
        counts += model_record.score()
    return counts[1] + counts[2]


class Record:

    def __init__(self: Record, statistics):
        """statistics is written by threshold_statistics.py"""
        self.fit_threshold, self.best_threshold = statistics[-2:]
        self.log_fit = numpy.log10(self.fit_threshold)


class Statistics:

    def __init__(self: Statistics, args):
        """Characteristics of training data for setting thresholds

        Args:
            args: Use args.records and args.statistics_path
        """

        with open(args.statistics_path, 'rb') as _file:
            record_names, fit, statistics = pickle.load(_file)

        self.record = {}
        for name in args.records:
            self.record[name] = Record(statistics[name])

    def f_mb_log_fit(
        self: Statistics,
        m: float,
        b: float,
        log_fit: float,
    ):
        """Map function of fit to threshold

        Args:
            m: Slope
            b: Intercept
            log_fit: Value of affine function

        Values of m, b are log_10 of thresholds
        """
        log_result = m * log_fit + b
        return 10**log_result

    def f_mb_name(self: Statistics, m: float, b: float, name):
        """Map name of record to threshold

        Args:
            m: Slope
            b: Intercept
            name: Name of record

        Values of m, b are log_10 of thresholds
        """
        return self.f_mb_log_fit(m, b, self.record[name].log_fit)


def main(argv=None):
    """Plot errors on training data against function parameters

    """

    # For detail of minimum
    m_s = numpy.linspace(.93, .95, 21)  # Slope m
    b_s = numpy.linspace(0.003, 0.01, 30)  # y intercept b
    # For ~ quadratic looking minima
    #m_s = numpy.linspace(0.5, 1.5, 21)  # Slope m
    #b_s = numpy.linspace(-1, 1, 21)  # y intercept b

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    m, b = args.mb
    if args.records is None:
        args.records = args.a_names

    model_records = dict((name, utilities.ModelRecord(args.model_path, name))
                         for name in args.records)

    statistics = Statistics(args)

    errors = errors_mb(m, b, model_records, statistics)
    print(f'{errors=}')

    fig, (min_axes, m_axes, b_axes) = pyplot.subplots(nrows=3, figsize=(6, 8))

    # Plot the function f_mb
    z_s = numpy.linspace(-4, 4, 100)
    y = numpy.array(list(statistics.f_mb_log_fit(m, b, z) for z in z_s))
    min_axes.semilogy(z_s, y)

    # Scatter plot of records
    for name, model_record in model_records.items():
        min_threshold = model_record.best_threshold()[0]
        log_fit = statistics.record[name].log_fit
        min_axes.semilogy(
            log_fit,
            min_threshold,
            marker=f'${name}$',
            markersize=14,
            linestyle='None',
        )
        min_axes.set_xlabel('log_fit')
        min_axes.set_ylabel('Min Threshold')

    errors = list(errors_mb(m_, b, model_records, statistics) for m_ in m_s)
    m_axes.plot(m_s, errors)
    m_axes.set_xlabel('slope')
    m_axes.set_ylabel('error count')

    errors = list(errors_mb(m, b_, model_records, statistics) for b_ in b_s)
    b_axes.plot(b_s, errors)
    b_axes.set_xlabel('intercept')
    b_axes.set_ylabel('error count')

    # Put tight_layout after plots
    fig.tight_layout()
    fig.savefig(args.fig_path)

    if args.show:
        pyplot.show()

    return 0


def write_shift_statistics(argv=None):
    """Alternative to main that creates and pickles a Statistics instance
    """
    # 19 seconds
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        "Pickle statistics for calculating thresholds")
    utilities.common_arguments(parser)
    parser.add_argument('model_path', type=str, help="path to model")
    parser.add_argument('pickle', type=str, help="path to result")
    args = parser.parse_args(argv)
    utilities.join_common(args)

    assert len(args.records) > 5

    model_records = dict((name, utilities.ModelRecord(args.model_path, name))
                         for name in args.records)
    statistics = Statistics(model_records, args)
    with open(args.pickle, 'wb') as _file:
        pickle.dump(statistics, _file)
    return 0


if __name__ == "__main__":
    if sys.argv[0] == "write_shift_statistics.py":
        sys.exit(write_shift_statistics())
    sys.exit(main())
