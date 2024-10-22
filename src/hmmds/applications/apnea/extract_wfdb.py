"""extract_wfdb.py: Use wfdb code to read wfdb record from PhysioNet
and pickle it.  From Rules.mk:

--shorten 204 $(PHYSIONET_WFDB) a03er $@

The data files in a03er are shorter than claimed in a03er.hea
(3,134,796 vs 3,135,000).  That causes wfdb.rdrecord() to crash.  I used
pdb to find the actual length.

"""
# https://wfdb.readthedocs.io/en/latest/wfdb.html says fs is sampling frequency
from __future__ import annotations  # Enables, eg, (self: HMM,

import sys
import argparse
import pickle
import os.path

import wfdb


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Extract data from wfdb record')
    parser.add_argument('--shorten',
                        type=int,
                        help="Don't read last part of signal")
    parser.add_argument('input_dir', type=str, help='Path to input dir')
    parser.add_argument('record_name', type=str, help='EG, a03er')
    parser.add_argument('output', type=str, help='Path to output')
    return parser.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    in_path = os.path.join(args.input_dir, args.record_name)
    header = wfdb.rdheader(in_path)
    length = header.sig_len
    if args.shorten:
        length -= args.shorten
    record = wfdb.rdrecord(in_path, sampto=length)
    result_dict = {}
    for name, signal in zip(record.sig_name, record.p_signal.T):
        result_dict[name] = signal
    result_dict['sample_frequency'] = (header.fs, 'Hz')
    result_dict['units'] = header.units

    with open(args.output, mode='wb') as _file:
        pickle.dump(result_dict, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
