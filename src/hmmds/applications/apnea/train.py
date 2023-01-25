"""train.py Command line specifies type of data and record names of data

Example:
python train.py --records a01 x02 b01 c05 --type ecg models/initial_ECG models/trained_ECG

The type selects one of the registered functions in this module.

"""
import sys
import os.path
import pickle
import argparse

import numpy

import hmm.base

import hmmds.applications.apnea.utilities
import hmmds.applications.apnea.observation
import develop


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Train an HMM on specified records")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('--records', type=str, nargs='+', help='EG: a01 x02')
    parser.add_argument('--type',
                        type=str,
                        help='A type registered in this module, eg, "ECG"')
    parser.add_argument('input', type=str, help='path to initial model')
    parser.add_argument('output', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


TYPES = {}  # Is populated by @register decorated functions.  The keys
# are function names, and the values are functions


def register(func):
    """Decorator that puts function in TYPES dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    TYPES[func.__name__] = func
    return func


@register
def ECG(args) -> develop.HMM:
    """A model for raw ecg data
    """

    # ToDo: Put this in utilities?
    def read_ecg(path):
        with open(path, 'rb') as _file:
            return pickle.load(_file)['raw']

    # Read the records
    paths = [os.path.join(args.rtimes, f'{name}.ecg') for name in args.records]
    return [read_ecg(path) for path in paths]


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    rng = numpy.random.default_rng()

    with open(args.input, 'rb') as _file:
        hmm = pickle.load(_file)

    # Use the registered function to read the training data
    data = TYPES[args.type](args)

    hmm.multi_train(data, args.iterations)

    with open(args.output, 'wb') as _file:
        pickle.dump(hmm, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
