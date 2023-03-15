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


# ToDo: Put this in utilities?
def read_ecg(path):
    with open(path, 'rb') as _file:
        return pickle.load(_file)['raw']

TYPES = {}  # Is populated by @register decorated functions.  The keys
# are function names, and the values are functions for reading data.

def register(func):
    """Decorator that puts function in TYPES dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    TYPES[func.__name__] = func
    return func

@register
def AR1k(args) -> develop.HMM:
    """Read raw ecg data for hmms with AR1k output models
    """

    # Read the records
    paths = [os.path.join(args.rtimes, f'{name}.ecg') for name in args.records]
    return [read_ecg(path) for path in paths]

@register
def AR3_(args) -> develop.HMM:
    """Read raw ecg data for hmms with AR3_ output models.  Sample
    data with segments a half hour apart to enable parallel training.

    """
    period = 10*60*100  # 100 samples per second
    interval = period//10  # Length of each segment
    paths = [os.path.join(args.rtimes, f'{name}.ecg') for name in args.records]
    result = []
    for path in paths:
        data = read_ecg(path)
        for start in range(0,int(len(data)),period):
            stop = min(start+interval, len(data))
            result.append(data[start:stop])
    return result

@register
def Masked(args):
    """Read a01 apply mask and return a 32 minute subset broken into
    16 intervals

    """
    n_minute = 60*100
    a01_masked_data = utilities.read_masked_ecg("a01", args)
    result = []
    for start_minute in range(75, 107 ,2):
        start = start_minute * n_minute
        stop = (start_minute+2) * n_minute
        print(f"{start=} {a01_masked_data.bundles[start:start+10]=}")
        result.append(a01_masked_data[start:stop])
    return result

@register
def AR3A_(args) -> develop.HMM:
    """A model for raw ecg data
    """
    return AR3_(args)

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
