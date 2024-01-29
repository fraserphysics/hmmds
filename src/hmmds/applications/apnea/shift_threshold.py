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


class Statistics:

    def __init__(self, record_names, args):
        """Calaulate characteristics of training data for setting thresholds
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
        for record_name in record_names:
            model_record = utilities.ModelRecord(args.model_path, record_name)
            pass1 = utilities.Pass1(model_record.record_name, args)
            n_a = model_record.class_from_expert.sum()
            log_t = numpy.log10(find_threshold(model_record, n_a)[0])
            self.log_threshold[record_name] = log_t
            psd = pass1.psd

            sum_lt += log_t
            sum_lt_sq += log_t * log_t
            sum_lt_psd += log_t * psd
            sum_psd += psd
            sum_psd_sq += psd * psd

        n_records = len(record_names)
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
        return result, z[channel_low:channel_high].sum()


def main(argv=None):
    """Plot best thresholds on training data against various statistics

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    if args.records is None:
        records = args.a_names
    else:
        records = args.records

    not7 = set(records) - set(('a07',))
    statistics = Statistics(records, args)
    fig, axeses = pyplot.subplots(nrows=4, figsize=(6, 4), sharex=False)

    bands = ((0, 60), (3, 60), (0, 50), (3, 50))
    for name in records:
        model_record = utilities.ModelRecord(args.model_path, name)
        for axes, band in zip(axeses, bands):
            f_psd, z = statistics.analyze(model_record, *band)
            axes.plot(
                f_psd,
                statistics.log_threshold[name],
                marker=f'${name}$',
                markersize=14,
                linestyle='None',
                color='r',
            )
            axes.plot(
                -z,
                statistics.log_threshold[name],
                marker=f'${name}$',
                markersize=14,
                linestyle='None',
                color='b',
            )
            axes.set_xlabel(f'{band}')
            axes.set_ylabel('Threshold')

    fig.savefig(args.fig_path)

    if args.show:
        pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
