"""pass2.py Makes a file that looks like the expert, ie,

python pass2.py --pass1 pass1.out --names a01 b01 c01 x01


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
import os
import argparse
import glob
import pickle

import numpy

import hmmds.applications.apnea.utilities
import develop


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser("Create and write pass2_report")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('--names',
                        type=str,
                        nargs='+',
                        help='names of records to analyze')
    parser.add_argument(
        '--model_paths',
        type=str,
        nargs=2,
        default=('../../../../build/derived_data/apnea/models/c_model',
                 '../../../../build/derived_data/apnea/models/two_ar6_masked6'),
        help='paths to model for records classified as N and A by pass1')
    parser.add_argument('pass1',
                        nargs='?',
                        type=argparse.FileType('rb'),
                        default='pass1.out',
                        help='Path to pass1 result')
    parser.add_argument('result',
                        nargs='?',
                        type=argparse.FileType('w', encoding='UTF-8'),
                        default=sys.stdout,
                        help='Write result to this path')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def analyze(name, model_path, report, debug=False):
    """Writes to the open file report a string that has the same form as
        the expert file

    Args:
        name: Eg, 'a01'
        model_path: A pickled HMM
        report: A file open for writing

    """

    model_record = hmmds.applications.apnea.utilities.ModelRecord(
        model_path, name)
    model_record.classify()
    model_record.score()

    model_record.formatted_result(report, expert=False)
    if not debug:
        return
    print('HMM')
    model_record.formatted_result(sys.stdout, expert=False)
    print('Expert')
    model_record.formatted_result(sys.stdout, expert=True)


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    pass1_dict = pickle.load(args.pass1)

    if not args.names:
        args.names = args.all_names

    model_paths = {'N': args.model_paths[0], 'A': args.model_paths[1]}

    for name in args.names:
        analyze(name, model_paths[pass1_dict[name]], args.result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
