"""survey_threshold.py  Study dependence of classification on
detection threshold.


"""
import sys
import argparse

import numpy

import hmmds.applications.apnea.utilities
import plotscripts.utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Map (model,data):-> class sequence")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('--thresholds',
                        type=str,
                        nargs=3,
                        help='Start, stop, number for range to evaluate')
    parser.add_argument('--expert_override',
                        type=str,
                        help="path to expert annotations")
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('model_path', type=str, help="path to initial model")
    parser.add_argument('fig_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def _print(text, results):
    print(f'{text:>9s} \
{"N_(N->A)":>8s} \
{"N_(A->N)":>8s} \
{"N_error":>8s} \
{"P_error":>10s}')
    for key, result in results.items():
        print(f"{key:>9.4g} \
{result['N false alarm']:>8d} \
{result['N missed detection']:>8d} \
{result['error count']:>8d} \
{result['error rate']:>10.5f}")


def log_plot(axes, results, xlabel=None):
    x = []
    error_rate = []
    false_alarm = []
    missed_detection = []
    for threshold, result in results.items():
        x.append(threshold)
        false_alarm.append(result['N false alarm'])
        missed_detection.append(result['N missed detection'])
        error_rate.append(result['error count'])
    axes.semilogx(x, false_alarm, label="false alarm")
    axes.semilogx(x, missed_detection, label="missed detection")
    axes.semilogx(x, error_rate, label="all errors")
    axes.legend()
    if xlabel:
        axes.set_xlabel(xlabel)
    axes.set_ylabel('Number of errors')


def plot(axes, results, xlabel=None):
    x = []
    error_rate = []
    false_alarm = []
    missed_detection = []
    for threshold, result in results.items():
        x.append(threshold)
        false_alarm.append(result['N false alarm'])
        missed_detection.append(result['N missed detection'])
        error_rate.append(result['error count'])
    axes.plot(x, false_alarm, label="false alarm")
    axes.plot(x, missed_detection, label="missed detection")
    axes.plot(x, error_rate, label="all errors")
    axes.legend()
    if xlabel:
        axes.set_xlabel(xlabel)
    axes.set_ylabel('Number of errors')


def threshold_study(model_record_dict, thresholds, power_dict):
    """Calculate errors as a function of thresholds
    """
    result = {}
    for threshold in thresholds:
        counts = numpy.zeros(4, dtype=int)
        for _, model_record in model_record_dict.items():
            model_record.classify(threshold, power_dict)
            counts += model_record.score()
        result[threshold] = {
            'N false alarm': counts[1],
            'N missed detection': counts[2],
            'P false alarm': counts[1] / (counts[0] + counts[1]),
            'P missed detection': counts[2] / (counts[2] + counts[3]),
            'error count': counts[1] + counts[2],
            'error rate': (counts[1] + counts[2]) / counts.sum(),
        }
    return result


def main(argv=None):
    """
    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    fig, axes = pyplot.subplots(nrows=1, figsize=(6, 4))

    def linspace(triple):
        return numpy.linspace(float(triple[0]), float(triple[1]),
                              int(triple[2]))

    exponents = linspace(args.thresholds)
    thresholds = numpy.exp(exponents * numpy.log(10))

    if args.records is None:
        records = args.a_names
    else:
        records = args.records

    # Ugly hack to let pickle.load in ModelRecord work
    sys.modules['utilities'] = hmmds.applications.apnea.utilities

    model_record_dict = {}
    for record_name in records:
        model_record_dict[
            record_name] = hmmds.applications.apnea.utilities.ModelRecord(
                args.model_path,
                record_name,
                heart_rate_path_format=args.heart_rate_path_format,
                expert=args.expert_override)

    threshold_results = threshold_study(model_record_dict, thresholds,
                                        args.power_dict)
    _print('threshold', threshold_results)

    log_plot(axes, threshold_results, xlabel='threshold')

    if args.show:
        pyplot.show()
    fig.tight_layout()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
