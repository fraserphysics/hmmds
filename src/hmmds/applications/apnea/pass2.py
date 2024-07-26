"""pass2.py Makes a file that looks like the expert, ie,

python pass2.py model pass2.out --records a01 b01 c01 x01


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


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser("Create and write pass2_report")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument(
        '--viterbi',
        action='store_true',
        help='Use Viterbi decoding instead of state probabilities')
    parser.add_argument('model_path', type=str, help='path to HMM')
    parser.add_argument('result',
                        nargs='?',
                        type=argparse.FileType('w', encoding='UTF-8'),
                        default=sys.stdout,
                        help='Write result to this path')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def analyze(name,
            model_path,
            report,
            debug=False,
            threshold=None,
            viterbi=False):
    """Writes to the open file report a string that has the same form as
        the expert file

    Args:
        name: Eg, 'a01'
        model_path: A pickled HMM
        report: A file open for writing
        threshold: Threshold for detector
        viterbi: Use viterbi decoding
    """

    model_record = hmmds.applications.apnea.utilities.ModelRecord(
        model_path, name)
    if viterbi:
        model_record.decode()
    else:
        model_record.classify(threshold)
    model_record.score()

    model_record.formatted_result(report, expert=False)
    if not debug:
        return
    print(f'HMM {threshold=}')
    model_record.formatted_result(sys.stdout, expert=False)
    print('Expert')
    model_record.formatted_result(sys.stdout, expert=True)


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    if not args.records:
        args.records = args.test_names

    for name in args.records:
        analyze(name,
                args.model_path,
                args.result,
                threshold=args.threshold,
                viterbi=args.viterbi,
                debug=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
