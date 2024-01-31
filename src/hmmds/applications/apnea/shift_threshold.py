"""shift_threshold.py Exploratory plots of dependence of threshold and N_A on F(PSD)

python shift_threshold.py [options] model_name [record_names]

"""
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
        "Plot best threshold and N_A against F(PSD), the pass1 statistic")
    utilities.common_arguments(parser)
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--fig_path',
                        type=str,
                        help="path to result",
                        default="shift_threshold.pdf")
    parser.add_argument('model_path', type=str, help="path to initial model")
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def for_threshold(threshold: float, model_record: utilities.ModelRecord,
                  reference: int):
    """Calculate and return (N_Apnea - reference) and counts
    Args:
        threshold: Detection threshold
        model_record: ModelRecord instance
        reference: Number of actual minutes marked apnea
    """
    model_record.classify(threshold)
    counts = model_record.score()
    value = counts[1] + counts[3] - reference
    return value, counts


def find_threshold(model_record: utilities.ModelRecord, n_apnea: int) -> float:
    """Find and return threshold that gets the correct number of
    apnea classifications

    """

    t_i = 1.0
    # Exponential search for bracket of f = 0
    f_1 = for_threshold(t_i, model_record, n_apnea)[0]
    if f_1 == 0:
        return t_i, 0
    elif f_1 > 0:  # too much apnea; increase threshold
        factor = 2.0
    else:
        factor = 1.0 / 2.0
    for i in range(100):
        t_2 = t_i * factor**i
        f_2 = for_threshold(t_2, model_record, n_apnea)[0]
        if f_2 * f_1 < 0:
            break

    # Set up and call brentq
    t_a = min(t_2 / factor, t_2)
    t_b = max(t_2 / factor, t_2)

    def f(log, args_0, args_1):
        """args_0 and args_1 are record_dict and n_apnea
        respectively.

        """
        return for_threshold(10**log, args_0, args_1)[0]

    l_0 = scipy.optimize.brentq(f,
                                numpy.log10(t_a),
                                numpy.log10(t_b),
                                args=(model_record, n_apnea),
                                rtol=5.0e-5)
    t_0 = 10**l_0
    f_0 = for_threshold(t_0, model_record, n_apnea)[0]
    return t_0, f_0


def f_abcd(z, x_0, x_1, y_0=.6, y_1=1300):
    if z < x_0:
        return y_0
    if z > x_1:
        return y_1
    slope = numpy.log(y_1 / y_0) / (x_1 - x_0)
    log_result = (z - x_0) * slope + numpy.log(y_0)
    return numpy.exp(log_result)


def minimum_error(model_record):
    """Find rough approximation of threshold that minimizes error for
    a record.  Result is only used for plotting.

    """
    thresholds = numpy.geomspace(.6, 1500, 10)
    result = numpy.zeros(len(thresholds), dtype=int)
    for i, threshold in enumerate(thresholds):
        model_record.classify(threshold=threshold)
        counts = model_record.score()
        result[i] = counts[1] + counts[2]
    return thresholds[numpy.argmin(result)]


def errors_abcd(a, b, c, d, model_records, statistics):
    """Report number of errors for thresholds function a,b

    Args:
        a: Threshold = .1 for x < a
        b: Threshold = 2000 for x > b
        model_records: List of ModelRecord instances
    """
    counts = numpy.zeros(4, dtype=int)
    for model_record in model_records.values():
        z = statistics.analyze(model_record, 0, 60)[1]
        threshold = f_abcd(-z, a, b, c, d)
        model_record.classify(threshold=threshold)
        counts += model_record.score()
    return counts[1] + counts[2]


class Statistics:

    def __init__(self, model_records, args):
        """Calaulate characteristics of training data for setting thresholds

        Args:
            model_records: EG, model_records['a01'] is a ModelRecord instance
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

    def analyze(self, model_record, low, high):
        """
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

    # Don't change b=1700.  Change d instead
    a, b, c, d = -1500, 1700, 11, 680
    a_s = numpy.linspace(-2000, -1000, 20)
    b_s = numpy.linspace(1600, 1800, 10)
    c_s = numpy.geomspace(1.0, 100, 20)
    d_s = numpy.geomspace(1e2, 2.0e3, 20)

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    if args.records is None:
        records = args.a_names
    else:
        records = args.records

    model_records = dict((name, utilities.ModelRecord(args.model_path, name))
                         for name in records)

    statistics = Statistics(model_records, args)  # 11 seconds

    errors = errors_abcd(a, b, c, d, model_records, statistics)
    print(f'{errors=}')

    fig, (min_axes, a_axes, b_axes, c_axes,
          d_axes) = pyplot.subplots(nrows=5, figsize=(6, 12))

    # Plot the function f_abcd
    x = numpy.linspace(-5800, 2500, 100)
    y = numpy.array(list(f_abcd(z, a, b, c, d) for z in x))
    min_axes.semilogy(x, y)

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
    sys.exit(main())
