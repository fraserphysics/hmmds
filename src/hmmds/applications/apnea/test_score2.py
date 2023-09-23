""" test_score2.py Test code that estimates class (normal vs apnea)

"""
import sys
import argparse

import numpy

import utilities
import plotscripts.utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Map (model,data):-> class sequence")
    utilities.common_arguments(parser)
    # FixMe: Use --records from utilities
    parser.add_argument('--record_name',
                        type=str,
                        default='a03',
                        help="eg, a03")
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('model_path', type=str, help="path to initial model")
    #parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def _print(results):
    for threshold, result in results.items():
        print(f"{threshold:8.4g} \
{result['false alarm']:11.4g} \
{result['missed detection']:16.4g} \
{result['error rate']:10.5f}")


def log_plot(axes, results, xlabel=None):
    x = []
    error_rate = []
    false_alarm = []
    missed_detection = []
    for threshold, result in results.items():
        x.append(threshold)
        false_alarm.append(result['false alarm'])
        missed_detection.append(result['missed detection'])
        error_rate.append(result['error rate'])
    axes.semilogx(x, false_alarm, label="false alarm")
    axes.semilogx(x, missed_detection, label="missed detection")
    axes.semilogx(x, error_rate, label="all errors")
    axes.legend()
    if xlabel:
        axes.set_xlabel(xlabel)


def plot(axes, results, xlabel=None):
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


def threshold_study(scores, thresholds, power):
    """Calculate errors as a function of thresholds
    """
    result = {}
    print(
        f'{"threshold":8s} {"false alarm":11s} {"missed detection":16s} {"error rate":10s}'
    )
    for threshold in thresholds:
        counts = numpy.zeros(4, dtype=int)
        for key, score in scores.items():
            score.score(threshold, power)
            counts += score.counts
        result[threshold] = {
            'false alarm': counts[1] / (counts[0] + counts[1]),
            'missed detection': counts[2] / (counts[2] + counts[3]),
            'error rate': (counts[1] + counts[2]) / counts.sum(),
        }
    return result


def power_study(scores, powers, threshold):
    """Calculate errors as a function of power on likelihood of
    interval component of observation

    """
    result = {}
    print(
        f'{"power":8s} {"false alarm":11s} {"missed detection":16s} {"error rate":10s}'
    )
    for power in powers:
        counts = numpy.zeros(4, dtype=int)
        for key, score in scores.items():
            score.score(threshold, power)
            counts += score.counts
        result[power] = {
            'false alarm': counts[1] / (counts[0] + counts[1]),
            'missed detection': counts[2] / (counts[2] + counts[3]),
            'error rate': (counts[1] + counts[2]) / counts.sum(),
        }
    return result


def print_expert_model(score):
    score.score(1.0)
    print('Expert')
    score.formatted_result(sys.stdout, expert=True)
    print('From Model')
    score.formatted_result(sys.stdout, expert=False)


def main(argv=None):
    """
    """

    min_power = 1.757
    powers = numpy.linspace(0.5, 2.0, 21)
    min_threshold = .73
    thresholds = numpy.exp(numpy.linspace(-15.0, 1.0, 21) * numpy.log(10))

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    fig, axeses = pyplot.subplots(nrows=2, figsize=(6, 8))

    scores = {}
    for record_name in args.a_names:
        scores[record_name] = utilities.Score2(args.model_path, record_name)

    power_results = power_study(scores, powers, min_threshold)
    _print(power_results)
    plot(axeses[1], power_results, xlabel='power')

    threshold_results = threshold_study(scores, thresholds, min_power)
    _print(threshold_results)
    log_plot(axeses[0], threshold_results, xlabel='threshold')

    if args.show:
        pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
