"""apnea_train.py reads an initial model trains it and writes the result

"""
import sys
import glob
import os.path
import pickle
import argparse

import numpy.random

import hmmds.applications.apnea.utilities
import hmmds.applications.apnea.model_init
import hmm.base
from hmmds.applications.apnea.utilities import State


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Read initial model, train, write result")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('--type',
                        type=str,
                        default="masked",
                        help='Type of data, eg, "masked" or "unmasked"')
    parser.add_argument('initial_path', type=str, help="path to initial model")
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


TYPES = {}  # Is populated by @register decorated functions.  The keys
# are function names, and the values are functions for reading data.


def register(func):
    """Decorator that puts function in TYPES dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    TYPES[func.__name__] = func
    return func


@register
def masked(args):
    return [
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_class(args, record))
        for record in args.records
    ]


@register
def unmasked(args):
    return [
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow(args, record))
        for record in args.records
    ]


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.initial_path, 'rb') as _file:
        old_args, model = pickle.load(_file)

    y_data = TYPES[args.type](args)

    model.multi_train(y_data, args.iterations)

    model.strip()
    with open(args.write_path, 'wb') as _file:
        pickle.dump((old_args, model), _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
