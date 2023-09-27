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
    parser.add_argument('--report_by_record',
                        action='store_true',
                        help="print results for each record")
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
    for prominence, scores_prominence in scores.items():
        counts = numpy.zeros(4, dtype=int)
        likelihoods = {}
        for record_name, score in scores_prominence.items():
            likelihoods[record_name] = score.score(best_threshold, best_power)
            if likelihoods[
                    record_name] > 0:  # Force detection of apnea for every minute
                score.counts[1] += score.counts[0]  # n2a += n2n
                score.counts[0] = 0
                score.counts[3] += score.counts[2]  # a2a += a2n
                score.counts[2] = 0
                score.class_from_model[:] = 1
            counts += score.counts
        result[prominence] = {
            'false alarm': counts[1] / (counts[0] + counts[1]),
            'missed detection': counts[2] / (counts[2] + counts[3]),
            'error rate': (counts[1] + counts[2]) / counts.sum(),
            'likelihoods': likelihoods,
        }
    return result


def print_by_record(scores, results):
    for prominence, scores_prominence in scores.items():
        results_p = results[prominence]
        print(f'{prominence=} error rate: {results_p["error rate"]}')
        print(
            f'{"record":6s} {"false alarm":11s} {"missed detection":16s} {"error rate":10s} {"likelihood":11s}'
        )

        names = list(scores_prominence.keys())

        def error_rate(name):
            score = scores_prominence[name]
            return (score.counts[1] + score.counts[2]) / score.counts.sum()

        names.sort(key=lambda x: -error_rate(x))
        for record_name in names:
            score = scores_prominence[record_name]
            false_alarm = score.counts[1] / (score.counts[0] + score.counts[1])
            missed_detection = score.counts[2] / (score.counts[2] +
                                                  score.counts[3])
            likelihood = results_p['likelihoods'][record_name]
            print(
                f'{record_name:>6s} {false_alarm:11.3f} {missed_detection:16.3f} {error_rate(record_name):10.3f} {likelihood:11.3g}'
            )


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

    if args.records is None:
        records = args.a_names
    else:
        records = args.records

    scores = {}
    for prominence in args.prominences:
        model_path = args.template.replace('%', prominence)
        float_key = float(prominence)
        scores[float_key] = {}
        for record_name in records:
            scores[float_key][record_name] = utilities.Score2(
                model_path, record_name)
    results = calculate(scores, best_threshold, best_power)
    if args.report_by_record:
        print_by_record(scores, results)
    print(
        f'{"peak threshold":14s} {"false alarm":11s} {"missed detection":16s} {"error rate":10s}'
    )
    _print(results)
    _plot(axeses[0], results, 'threshold prominence')
    fig.savefig(args.fig_path)

    if args.show:
        pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
