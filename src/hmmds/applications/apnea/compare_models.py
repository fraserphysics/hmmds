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

import utilities
import plotscripts.utilities


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
    parser.add_argument('--model_dir',
                        type=str,
                        default='../../../../build/derived_data/apnea/models')
    parser.add_argument('--parameter_name', type=str, default='parameter')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--fig_path', type=str, help="path to result")
    parser.add_argument('--latex', type=str, help="resulting latex table")
    parser.add_argument('--result',
                        type=argparse.FileType('w', encoding='UTF-8'),
                        default=sys.stdout,
                        help='Write result to this path')
    parser.add_argument('figure_path',
                        nargs='?',
                        type=str,
                        default='compare_models.pdf',
                        help='Write result to this path')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def print_summary(results):
    print(
        f'{"peak threshold":14s} {"false alarm":11s} {"missed detection":16s} {"error rate":10s}'
    )
    for threshold, result in results.items():
        print(f"{threshold:14.4g} \
{result['false alarm']:11.4g} \
{result['missed detection']:16.4g} \
{result['error rate']:10.5f}")


def _plot(axes, results, xlabel=None):
    x = []
    error_rate = []
    false_alarm = []
    missed_detection = []
    for threshold, result in results.items():
        x.append(threshold)
        false_alarm.append(result['false alarm'])
        missed_detection.append(result['missed detection'])
        error_rate.append(result['error rate'])
    axes.plot(x, false_alarm, label="false alarm")
    axes.plot(x, missed_detection, label="missed detection")
    axes.plot(x, error_rate, label="all errors")
    axes.legend()
    if xlabel:
        axes.set_xlabel(xlabel)


def main(argv=None):
    """Plot pass2 classification performance against minimum peak
    prominence

    """

    best_power = 4.8  # Raise likelihood of interval to this power
    best_threshold = 2.0e-43  # Threshold of apnea detector

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    fig, axes = pyplot.subplots(nrows=1, figsize=(6, 8))

    if args.records is None:
        records = args.a_names
    else:
        records = args.records

    assert len(args.models) == len(args.parameters)
    error_rates = []
    for model_name, parameter in zip(args.models, args.parameters):
        counts = numpy.zeros(4)
        model_path = os.path.join(args.model_dir, model_name)
        for record_name in records:
            instance = utilities.Score2(model_path, record_name)
            instance.score(best_threshold, best_power)
            counts += instance.counts
        error_rates.append((counts[1] + counts[2]) / counts.sum())
        print(
            f'{parameter} N->A: {int(counts[1])} A->N: {int(counts[2])} N_error: {int(counts[1] + counts[2])} P_error: {error_rates[-1]:5.3f}'
        )
    axes.plot(args.parameters, error_rates)
    axes.set_xlabel(args.parameter_name)
    axes.set_ylabel(r'$P_{\rm{Error}}$')

    fig.savefig(args.figure_path)

    if args.show:
        pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
