"""prominence_study.py Plot pass2 classification performance against
minimum peak prominence

"""
import sys
import argparse
import typing

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
    parser.add_argument('--fig_path', type=str, help="path to result")
    parser.add_argument('--latex', type=str, help="resulting latex table")
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def print_summary(results):
    print(
        f'{"peak threshold":14s} {"false alarm":11s} {"missed detection":16s} {"error count":11s}{"error rate":10s}'
    )
    for threshold, result in results.items():
        print(f"{threshold:14.4g} \
{result['false alarm']:11.4g} \
{result['missed detection']:16.4g} \
{result['error count']:11d} \
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


def calculate(model_record_dict, best_threshold, best_power):
    """Calculate errors as a function of minimum prominence for peak detection
    """

    result = {}
    for prominence, dict_record in model_record_dict.items():
        counts = numpy.zeros(4, dtype=int)
        likelihoods = {}
        for record_name, model_record in dict_record.items():
            model_record.classify(best_threshold, best_power)
            likelihoods[record_name] = model_record.model.forward()
            counts += model_record.score()
        result[prominence] = {
            'false alarm': counts[1] / (counts[0] + counts[1]),
            'missed detection': counts[2] / (counts[2] + counts[3]),
            'error rate': (counts[1] + counts[2]) / counts.sum(),
            'error count': counts[1] + counts[2],
            'likelihoods': likelihoods,
        }
    return result


def print_by_record(model_record_dict,
                    results,
                    report: typing.TextIO,
                    latex=False):
    for prominence, dict_record in model_record_dict.items():
        results_p = results[prominence]
        if latex:
            print(r'''\begin{tabular}{|r|rr|rr|rr|}
\hline''', file=report)
            print(
                r'record & $N_N$ & $P_{FA}$ & $N_A$ &$P_{MD}$ & $P_{E}$ & $\log(p(y|\theta))$ \\',
                file=report)
        else:
            print(f'{prominence=} error rate: {results_p["error rate"]}',
                  file=report)
            print(
                f'{"record":6s} {"false alarm":11s} {"missed detection":16s} {"error rate":10s} {"likelihood":11s}',
                file=report)

        names = list(dict_record.keys())

        def error_rate(name):
            model_record = dict_record[name]
            return (model_record.counts[1] +
                    model_record.counts[2]) / model_record.counts.sum()

        names.sort(key=lambda x: -error_rate(x))
        for record_name in names:
            model_record = dict_record[record_name]
            n_normal = model_record.counts[0] + model_record.counts[1]
            false_alarm = model_record.counts[1] / n_normal
            n_apnea = model_record.counts[2] + model_record.counts[3]
            missed_detection = model_record.counts[2] / n_apnea
            likelihood = results_p['likelihoods'][record_name]
            if latex:
                print(
                    f'{record_name} & {n_normal:} & {false_alarm:5.3f} & {n_apnea} & {missed_detection:5.3f} & {error_rate(record_name):5.3f} & {likelihood:7.3g} \\\\',
                    file=report)
            else:
                print(
                    f'{record_name:>6s} {false_alarm:11.3f} {missed_detection:16.3f} {error_rate(record_name):10.3f} {likelihood:11.3g}',
                    file=report)
        if latex:
            print(r'\hline \end{tabular}', file=report)


def main(argv=None):
    """Plot pass2 classification performance against minimum peak
    prominence

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    fig, axeses = pyplot.subplots(nrows=2, figsize=(6, 8))

    best_power, best_threshold = args.power_and_threshold
    if args.records is None:
        records = args.a_names
    else:
        records = args.records

    model_record_dict = {}
    for prominence in args.prominences:
        model_path = args.template.replace('%', prominence)
        float_key = float(prominence)
        model_record_dict[float_key] = {}
        for record_name in records:
            model_record_dict[float_key][record_name] = utilities.ModelRecord(
                model_path, record_name)
    results = calculate(model_record_dict, best_threshold, best_power)
    if args.report_by_record:
        print_by_record(model_record_dict, results, sys.stdout)
    if args.latex is not None:
        with open(args.latex, encoding='utf-8', mode='w') as _file:
            print_by_record(model_record_dict, results, _file, latex=True)
    print_summary(results)

    # Cheap to plot.  Make it even if not used
    _plot(axeses[0], results, 'threshold prominence')
    if args.fig_path is not None:
        fig.savefig(args.fig_path)

    if args.show:
        pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
