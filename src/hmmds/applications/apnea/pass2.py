"""pass2.py Makes a file that looks like the expert, ie,

python pass2.py --records a01 b01 c01 x01 --statistics ../../../../build/derived_data/apnea/statistics64.pkl pass1.out pass2.out


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
    parser.add_argument('--c_threshold',
                        type=float,
                        default=4.0,
                        help='Threshold to make all N')
    parser.add_argument('--cheat',
                        action='store_true',
                        help='Use best threshold for each record')
    parser.add_argument(
        '--model_paths',
        type=str,
        nargs=2,
        default=
        ('../../../../build/derived_data/apnea/models/c_model',
         '../../../../build/derived_data/apnea/models/multi_ar12fs4lpp65rc13rw3.0rs.455_masked'
        ),
        help=
        'paths to models for records classified as N and A respectively by pass1'
    )
    parser.add_argument(
        '--statistics',
        type=argparse.FileType('rb'),
        default='../../../../build/derived_data/apnea/statistics64.pkl',
        help='Path to statistics for threshold')
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
        raise RuntimeError('Not implemented')
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

    pass1_dict = pickle.load(args.pass1)
    statistics = pickle.load(args.statistics)
    coefficients = statistics['threshold_coefficients']

    if not args.records:
        args.records = args.test_names

    model_paths = {'N': args.model_paths[0], 'A': args.model_paths[1]}

    for name in args.records:
        model_path = model_paths[pass1_dict[name]]
        if os.path.basename(model_path) == 'c_model':
            print(f'{name} {pass1_dict[name]}')
            analyze(name, model_path, args.result, threshold=args.c_threshold)
            continue
        psd = numpy.log10(statistics['psds'][name])
        log = min(2.5, max(-2.5, numpy.dot(coefficients, psd)))
        print(f'{name} {pass1_dict[name]} {log:5.2f}')
        analyze(name, model_path, args.result, threshold=10**log, debug=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
