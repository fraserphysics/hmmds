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
import utilities


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
# are function names, and the values are functions for reading data.


def register(func):
    """Decorator that puts function in TYPES dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    TYPES[func.__name__] = func
    return func


def segment(data):
    """Break ECG data into 15 minute segments for parallel processing

    """

    f_pm = 60 * 100  # Samples per minute
    minutes_per_segment = 15
    samples_per_segment = minutes_per_segment * f_pm

    result = []
    for n_start in range(0, len(data), samples_per_segment):
        n_stop = n_start + samples_per_segment
        result.append(data[n_start:n_stop])
    return result


@register
def AR3_(args) -> develop.HMM:
    """Read raw ecg data for hmms with AR3_ output models.  Sample
    data with 15 minute segments to enable parallel training.

    """
    paths = [os.path.join(args.rtimes, f'{name}.ecg') for name in args.records]
    result = []
    for path in paths:
        result.extend(segment(utilities.read_ecg(path)))
    return result


@register
def Dict(args):
    """Read a01, create classes, and return 32 subsets of 15 minute intervals.

    """
    n_before = 18  # Number of samples before peak
    n_after = 30  # Number of samples after peak
    a01_tagged = utilities.read_tagged_ecg("a01", args, n_before, n_after)
    return segment(a01_tagged)


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.input, 'rb') as _file:
        _hmm = pickle.load(_file)

    # Use the registered function to read the training data
    data = TYPES[(args.type).rstrip('0123456789')](args)

    _hmm.multi_train(data, args.iterations)
    _hmm.strip()

    with open(args.output, 'wb') as _file:
        pickle.dump(_hmm, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
