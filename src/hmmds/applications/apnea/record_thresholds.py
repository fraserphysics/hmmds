"""record_thresholds.py Best threshold for models for single records.

python record_thresholds.py $(MULTI_BEST) $(MODELS)/class record_thresholds.pkl

# record_thresholds['a01']() = best threshold for classifying data[a01] using
# class[a01], $(MULTI_BEST)

"""
from __future__ import annotations

import sys
import argparse
import typing
import pickle
import os

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
                        default=(1.0e-4, 1.0e4, 10),
                        help="geometric range of thresholds")
    parser.add_argument('best_model', type=str, help="path to model")
    parser.add_argument('class_model_dir', type=str, help="path to models")
    parser.add_argument('result_path', type=str, help="path to pickle file")
    args = parser.parse_args(argv)
    utilities.join_common(args)
    args.low = float(args.resolution[0])
    args.high = float(args.resolution[1])
    args.levels = int(args.resolution[2])
    return args


def main(argv=None):
    """Calculate various statistics and parameters for f(record) ->
    threshold, and write to a pickle file

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    model_names = os.listdir(args.class_model_dir)
    assert len(model_names) > 1

    result = {}
    for model_name in model_names:
        model_path = os.path.join(args.class_model_dir, model_name)
        model_record = utilities.ModelRecord(model_path, model_name)
        threshold_self = model_record.best_threshold(args.low, args.high,
                                                     args.levels)

        model_record = utilities.ModelRecord(args.best_model, model_name)
        threshold_best_model = model_record.best_threshold(
            args.low, args.high, args.levels)

        result[model_name] = (threshold_self, threshold_best_model)

        def print_one(pair):
            print(f'{pair[0]:8.2e} ', end='')
            for count in pair[1]:
                print(f'{count:3d} ', end='')
            print(f'{pair[1][1]+pair[1][2]:3d}# ', end='')

        print(f'{model_name} ', end='')
        print_one(threshold_self)
        print_one(threshold_best_model)
        print('')

    with open(args.result_path, 'wb') as _file:
        pickle.dump(result, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
