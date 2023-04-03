"""ecg_likekihood.py Calculate the sequence of conditional likelihoods of an ECG record

Example:
python ecg_likelihood.py trained_01 a01 states_a01

"""
import sys
import os.path
import pickle
import argparse

import numpy

import hmmds.applications.apnea.utilities
import hmm.C


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Estimate state sequence")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('hmm', type=str, help='Path to hmm')
    parser.add_argument('record', type=str, help='Name of record, eg, a01')
    parser.add_argument('output', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.hmm, 'rb') as _file:
        _, _hmm = pickle.load(_file)

    ecg_path = os.path.join(args.rtimes, args.record + ".ecg")
    with open(ecg_path, 'rb') as _file:
        _dict = pickle.load(_file)
        ecg = _dict["raw"]

    likelihood_sequence = _hmm.likelihood([ecg])

    with open(args.output, 'wb') as _file:
        pickle.dump(likelihood_sequence, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
