"""record_thresholds.py Print best threshold for models for single records.

python record_thresholds.py $(MULTI_BEST)

"""
from __future__ import annotations

import sys
import argparse
import typing

import numpy

import utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Fit function for threshold of each record")
    utilities.common_arguments(parser)
    parser.add_argument('--resolution',
                        type=float,
                        nargs=3,
                        default=(-4, 4, 9),
                        help="geometric range of thresholds")
    parser.add_argument('best_model', type=str, help="path to model")
    args = parser.parse_args(argv)
    utilities.join_common(args)
    args.low = 10**float(args.resolution[0])
    args.high = 10**float(args.resolution[1])
    args.levels = int(args.resolution[2])
    return args


def main(argv=None):
    """Calculate best threshold for each record in APLUSNAMES

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    if args.records:
        record_names = args.records
    else:
        record_names = args.a_names + 'b01 b02 b03 b04 c08 c10'.split()

    result = {}
    for record_name in record_names:

        model_record = utilities.ModelRecord(args.best_model, record_name)
        result[record_name] = numpy.log10(
            model_record.best_threshold(args.low, args.high, args.levels)[0])

    record_names.sort(key=lambda x: result[x])

    for record_name in record_names:
        print(f'{record_name} {result[record_name]:6.2f}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
