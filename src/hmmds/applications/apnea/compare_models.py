"""compare_models.py Calculate and plot error rate for list of models

python compare_models.py --models two_ar1_masked two_ar4_masked \
two_ar6_masked --parameters 1 4 6 --show

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
from shift_threshold import Statistics


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Plot error rate for list of models")
    utilities.common_arguments(parser)
    parser.add_argument('--models',
                        type=str,
                        nargs='+',
                        help="eg, ar_3 ar_4 ar_5 ar_6")
    parser.add_argument('--parameters',
                        type=float,
                        nargs='+',
                        help="eg, 3.0 4.0 5.0 6.0")
    parser.add_argument('--n_apnea',
                        type=int,
                        help='obtain by shifting threshold')
    parser.add_argument('--parameter_name', type=str, default='parameter')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--latex', type=str, help="resulting latex table")
    parser.add_argument('--result',
                        type=argparse.FileType('w', encoding='UTF-8'),
                        default=sys.stdout,
                        help='Write result to this path')
    parser.add_argument('--abcd',
                        type=float,
                        nargs=4,
                        help='Parameters of map: PSD -> Threshold')
    parser.add_argument('--shift_statistics',
                        type=str,
                        help='path to threshold shift statistics')
    parser.add_argument('figure_path',
                        nargs='?',
                        type=str,
                        default='compare_models.pdf',
                        help='Write result to this path')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def print_summary(error_counts):
    print(
        f'{"key":>10s}  {"N->N":5s} {"N->A":5s} {"A->N":5s} {"A->A":5s} {"N_err":5s} {"F_err":5s}'
    )
    for key, counts in error_counts.items():
        error_fraction = (counts[1] + counts[2]) / counts.sum()
        print(
            f'{key:10.3f} {counts[0]:5d} {counts[1]:5d} {counts[2]:5d} {counts[3]:5d} {counts[1] + counts[2]:5d}  {error_fraction:5.3f}'
        )
    print(
        f'total_apnea={counts[2]+counts[3]}, total_normal={counts[0]+counts[1]}'
    )


def plot(axes, error_counts, xlabel=None):
    x = []
    error_rate = []
    false_alarm = []
    missed_detection = []
    for x_, counts in error_counts.items():
        x.append(float(x_))
        false_alarm.append(counts[1])
        missed_detection.append(counts[2])
        error_rate.append(counts[1] + counts[2])
    axes.plot(x, false_alarm, label="false alarm")
    axes.plot(x, missed_detection, label="missed detection")
    axes.plot(x, error_rate, label="all errors")
    axes.legend()
    if xlabel:
        axes.set_xlabel(xlabel)
    axes.set_ylabel('Number of errors')


def plot_two(axeses, error_counts, thresholds, xlabel=None):
    error_axes, threshold_axes = axeses
    x = []
    error_rate = []
    threshold_list = []
    for x_, counts in error_counts.items():
        x.append(float(x_))
        error_rate.append(counts[1] + counts[2])
        threshold_list.append(thresholds[x_][0])
    error_axes.plot(x, error_rate, label="N errors")
    error_axes.set_ylabel('N errors')
    error_axes.legend()
    threshold_axes.semilogy(x, threshold_list, label="threshold")
    threshold_axes.legend()
    if xlabel:
        threshold_axes.set_xlabel(xlabel)
    threshold_axes.set_ylabel('Threshold')


def for_threshold(threshold, record_dict, reference=6446):
    """Calculate and return (N_Apnea - reference) and counts
    Args:
        threshold: Detection threshold
        record_dict:  keys are record names, values are ModelRecord instances
        reference: Number of actual minutes marked apnea
    """
    counts = numpy.zeros(4, dtype=int)
    for model_record in record_dict.values():
        model_record.classify(threshold)
        counts += model_record.score()
    value = counts[1] + counts[3] - reference
    return value, counts


def abcd_counts(abcd, statistics, record_dict):
    """Calculate classification counts with per record thresholds

    Args:
        abcd: Parameters of threshold function
        statistics: statistics.f_abcd is the threshold function
        record_dict: Keys are record names.  Values are ModelRecord instances
    """
    counts = numpy.zeros(4, dtype=int)
    for name, model_record in record_dict.items():
        threshold = statistics.f_abcd(*abcd, model_record=model_record)
        model_record.classify(threshold)
        counts += model_record.score()
    return counts


def find_threshold(record_dict, t_i, n_apnea):
    """ Solve for t: for_threshold(t,record_dict) = 0

    Args:
        record_dict: Keys are record names and values are ModelRecord instances.
        t_i: Initial guess for threshold
        n_apnea:
    """

    # Exponential search for bracket of f = 0
    f_1 = for_threshold(t_i, record_dict, n_apnea)[0]
    if f_1 == 0:
        return t_i, 0
    elif f_1 > 0:  # too much apnea; increase threshold
        factor = 2.0
    else:
        factor = 1.0 / 2.0
    for i in range(100):
        t_2 = t_i * factor**i
        f_2 = for_threshold(t_2, record_dict, n_apnea)[0]
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
                                args=(record_dict, n_apnea),
                                rtol=5.0e-5)
    t_0 = 10**l_0
    f_0 = for_threshold(t_0, record_dict, n_apnea)[0]
    return t_0, f_0


def main(argv=None):
    """Plot pass2 classification performance against minimum peak
    prominence

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    if args.records is None:
        records = args.a_names
    else:
        records = args.records

    assert len(args.models) == len(
        args.parameters), f'{args.models=} {args.parameters=}'

    if args.shift_statistics:
        assert args.abcd is not None
        with open(args.shift_statistics, 'rb') as _file:
            shift_statistics = pickle.load(_file)

    model_records = {}
    for model_name in args.models:
        model_path = os.path.join(args.model_dir, model_name)
        model_records[model_name] = {}
        for record_name in records:
            model_records[model_name][record_name] = utilities.ModelRecord(
                model_path, record_name)

    error_counts = {}
    if args.n_apnea:
        thresholds = {}
        t_0 = 1.0
        for model_name, parameter in zip(args.models, args.parameters):
            t_0, f_0 = find_threshold(model_records[model_name], t_0,
                                      args.n_apnea)
            thresholds[parameter] = (t_0, f_0)
            error_counts[parameter] = for_threshold(
                t_0, model_records[model_name])[1]

        fig, axeses = pyplot.subplots(nrows=2, figsize=(6, 4), sharex=True)
        plot_two(axeses, error_counts, thresholds, args.parameter_name)
    else:
        for model_name, parameter in zip(args.models, args.parameters):
            if args.abcd:
                error_counts[parameter] = abcd_counts(args.abcd,
                                                      shift_statistics,
                                                      model_records[model_name])
            else:
                error_counts[parameter] = for_threshold(
                    args.threshold, model_records[model_name])[1]

        fig, axes = pyplot.subplots(nrows=1, figsize=(6, 4))
        plot(axes, error_counts, args.parameter_name)
    print_summary(error_counts)

    fig.tight_layout()
    fig.savefig(args.figure_path)

    if args.show:
        pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
