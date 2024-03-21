"""neighbors_pass2.py Classify each minute of each record.

python neighbors_pass2.py pass1.out bare_likelihoods record_thresholds class_models result

For each record, eg. 'axx', find the record, eg. ayy, that maximizes
Prob(data[axx]|base_model[ayy]).  Then use class_model[ayy] with
record_threshold[ayy] to classify each minute in axx.

Write the result in the same format that as the expert training data,
ie,

a01
 0 NNAAAANNNNNNNNNNNNAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 1 AAAAAAAAAAAAAANNNNNNNNNNNNNNNNNAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 2 AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 3 AAAAAAAAAAAAAAAAAAAAAANNNNNNNNNNNNNNNNNNNNNNNNAAAAAAAAAAAAAA
 4 AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 5 NNAAAANNNNNNNNNNAAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAAAAAAAAAA
 6 AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 7 AAAAAAAAAAAAAAAAAAAAAAAAAAAANNNNAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 8 AAAAAAAAA
b01
 0 NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAAAAAAANNNNNNNNNNNNNNNNNN
.
.
.

"""
import sys
import argparse
import pickle
import os

import numpy

import hmmds.applications.apnea.utilities
import develop


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser("Create and write pass2_report")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('pass1',
                        type=argparse.FileType('rb'),
                        help='Path to pass1 result')
    parser.add_argument('bare_likelihoods',
                        type=argparse.FileType('rb'),
                        help='Path to pickled dict of likelihoods')
    parser.add_argument('record_thresholds',
                        type=argparse.FileType('rb'),
                        help='Path to pickled dict of thresholds')
    parser.add_argument('best', type=str, help='Path to best model')
    parser.add_argument('result',
                        type=argparse.FileType('w', encoding='UTF-8'),
                        default=sys.stdout,
                        help='Write result to this path')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def average_threshold(likelihoods, record_thresholds):
    """"""

    log_weights = numpy.empty(len(likelihoods))
    log_thresholds = numpy.empty(len(likelihoods))
    for index, (log_likelihood, name) in enumerate(likelihoods):
        log_weights[index] = log_likelihood
        log_thresholds[index] = numpy.log(record_thresholds[name][1][0])
    log_weights -= log_weights[0]
    log_weights /= 30
    weights = numpy.exp(log_weights - log_weights[0])
    weights /= weights.sum()
    threshold = numpy.exp(numpy.dot(weights, log_thresholds))
    print(
        f'''best threshold = {record_thresholds[likelihoods[0][1]][1][0]} average={threshold}
{weights[:3]=}''')
    return threshold


def analyze(record_name: str, likelihoods: dict, best: str,
            record_thresholds: dict, report):
    """Writes to "report" the analysis of record_name based on
    model[s] named in sorted list likelihoods.

    Args:
        record_name: Record to analyze
        likelihoods: Sorted list of pairs (name, log_likelihood)
        best: Directory to single model optimized for classification
        record_thresholds: record_thresholds['a01'] = (data for self, data for best_model)
        report: A file open for writing

    The data in record_thresholds is (threshold, counts).


    """

    threshold = average_threshold(likelihoods, record_thresholds)

    def print_(likelihood):
        print(f'{likelihood[1]} {likelihood[0]:8.1e}, ', end='')

    print(f'{record_name} ', end='')

    #closest = likelihoods[0][1]

    # record_thresholds[closest][0] is for self
    # record_thresholds[closest][0][0] is the threshold
    #threshold = record_thresholds[closest][1][0]
    #model_path = os.path.join(class_path, closest)

    model_record = hmmds.applications.apnea.utilities.ModelRecord(
        best, record_name)
    model_record.classify(threshold)
    model_record.score()
    model_record.formatted_result(report, expert=False)


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    #pass1_dict = pickle.load(args.pass1)
    bare_likelihoods = pickle.load(args.bare_likelihoods)
    record_thresholds = pickle.load(args.record_thresholds)

    if not args.records:
        args.records = args.all_names

    for record_name in args.records:
        likelihoods = []
        for model_name in record_thresholds.keys():
            if model_name == record_name:
                continue
            likelihoods.append(
                (bare_likelihoods[model_name][record_name], model_name))
        likelihoods.sort(key=lambda x: -x[0])
        analyze(record_name, likelihoods, args.best, record_thresholds,
                args.result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
