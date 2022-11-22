""" respire.py:  Extract respiration signature from low pass heart rate files

Imitates respire.py in my hmmds3 project
"""

import sys
import os.path
import argparse
import pickle

import pint
import numpy
import numpy.fft

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Calculate respiration from Rtimes')
    parser.add_argument('--samples_per_minute', type=int, default=10)
    parser.add_argument('--fft_width',
                        type=int,
                        default=1024,
                        help='Number of samples for each fft')
    parser.add_argument('--bound',
                        type=float,
                        help='Bound on samples used for calculating basis functions')
    parser.add_argument('--window_width',
                        type=float,
                        help='Sigma of Gaussian window for fft')_
    parser.add_argument('annotations', type=str, help='File of expert annotations')
    parser.add_argument('Rtimes_dir', type=str, help='Path to Rtimes data')
    parser.add_argument('resp_dir',
                        type=str,
                        help='Path to respiration data')
    return parser.parse_args(argv)

def read_rtimes(path):
    """Duplicate of function in rtimes2hr
    """
    with open(path, mode='r', encoding='utf-8') as _file:
        lines = _file.readlines()
    n_times = len(lines)
    rtimes = numpy.empty(n_times)
    for t, line in enumerate(lines):
        rtimes[t] = float(line)
    return rtimes * PINT('second')


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    # Read r_times data
    all_names = []  # Namely: ['a01', 'a02', ..., 'x35']
    records = {'a':{}, 'b':{}, 'c':{}, 'x':{}}
    for name in os.listdir(args.Rtimes_dir):
        assert name[0] in 'abcx'
        assert int(name[-2:]) > 0 # Ensure that name ends in digits
        # assign a01 data to records['a']['a01']
        pass

    # Do linear discriminant analysis and write summary

    # Calculate a respiration signal for each record and write the result
    for name in all_names:
        result = 'foo'
        with open(os.path.join(args.resp_dir, name + '.resp'), 'wb') as _file:
            pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
