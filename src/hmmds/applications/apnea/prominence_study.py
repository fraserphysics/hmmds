"""prominence_study.py Plot pass2 classification performance against
minimum peak prominence

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
    parser.add_argument('--prominences',
                        type=str,
                        nargs='+',
                        help="eg, 3.0 4.0 5.0 6.0")
    parser.add_argument('--template', type=str, help="eg, root/two_ar3_masked%")
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('fig_path', type=str, help="path to result")
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def _print(results):
    for threshold, result in results.items():
        print(f"{threshold:8.4g} \
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


def calculate(scores, best_threshold, best_power):
    """Calculate errors as a function of minimum prominence for peak detection
    """

    result = {}
    print(
        f'{"peak threshold":14s} {"false alarm":11s} {"missed detection":16s} {"error rate":10s}'
    )
    for prominence, value in scores.items():
        counts = numpy.zeros(4, dtype=int)
        for record_name, score in scores[prominence].items():
            score.score(best_threshold, best_power)
            counts += score.counts
        result[prominence] = {
            'false alarm': counts[1] / (counts[0] + counts[1]),
            'missed detection': counts[2] / (counts[2] + counts[3]),
            'error rate': (counts[1] + counts[2]) / counts.sum(),
        }
    return result


def main(argv=None):
    """Plot pass2 classification performance against minimum peak
    prominence

    """

    best_power = 1.757  # Raise likelihood of interval to this power
    best_threshold = .73  # Threshold of apnea detector

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    fig, axeses = pyplot.subplots(nrows=2, figsize=(6, 8))

    scores = {}
    for prominence in args.prominences:
        model_path = args.template.replace('%', prominence)
        float_key = float(prominence)
        scores[float_key] = {}
        for record_name in args.a_names:
            scores[float_key][record_name] = utilities.Score2(
                model_path, record_name)
    results = calculate(scores, best_threshold, best_power)
    _print(results)
    _plot(axeses[0], results, 'threshold prominence')
    fig.savefig(args.fig_path)

    if args.show:
        pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
