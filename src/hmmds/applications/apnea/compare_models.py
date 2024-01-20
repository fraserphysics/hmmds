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


def plot(axes, results, xlabel=None):
    x = []
    error_rate = []
    false_alarm = []
    missed_detection = []
    for x_, result in results.items():
        x.append(x_)
        false_alarm.append(result['N false alarm'])
        missed_detection.append(result['N missed detection'])
        error_rate.append(result['error count'])
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

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    fig, axes = pyplot.subplots(nrows=1, figsize=(6, 4))

    threshold = args.threshold

    if args.records is None:
        records = args.a_names
    else:
        records = args.records

    assert len(args.models) == len(
        args.parameters), f'{args.models=} {args.parameters=}'
    error_rates = {}
    for model_name, parameter in zip(args.models, args.parameters):
        counts = numpy.zeros(4)
        model_path = os.path.join(args.model_dir, model_name)
        for record_name in records:
            instance = utilities.ModelRecord(model_path, record_name)
            instance.classify(threshold)
            counts += instance.score()
        error_rates[parameter] = {
            'N false alarm': counts[1],
            'N missed detection': counts[2],
            'P false alarm': counts[1] / (counts[0] + counts[1]),
            'P missed detection': counts[2] / (counts[2] + counts[3]),
            'error count': counts[1] + counts[2],
            'error rate': (counts[1] + counts[2]) / counts.sum(),
        }
        print(
            f'{parameter} N->A: {int(counts[1])} A->N: {int(counts[2])} N_error: {int(counts[1] + counts[2])} P_error: {error_rates[parameter]["error rate"]:5.3f}'
        )
    plot(axes, error_rates, args.parameter_name)

    fig.savefig(args.figure_path)

    if args.show:
        pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
