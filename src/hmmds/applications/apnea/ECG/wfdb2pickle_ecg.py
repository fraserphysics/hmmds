""" wfdb2pickle_ecg.py: Use wfdb code to read wfdb records from PhysioNet
and write ecg information in pickle format.

python wfdb2pickle_ecg.py wfdb_dir ecg_dir a01 a02 b01 b02 x35

"""
# https://wfdb.readthedocs.io/en/latest/wfdb.html says fs is sampling frequency
from __future__ import annotations  # Enables, eg, (self: HMM,

import sys
import argparse
import pickle
import os.path
import typing

import numpy

import wfdb
import wfdb.processing

# define expected record parameters
sample_units = ['mV']
sample_frequency = 100  # Hz


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Translate ecgs from wfdb to pickle')
    parser.add_argument('wfdb_dir', type=str, help='Path to ecg data')
    parser.add_argument('ecg_dir', type=str, help='Path to ECG data')
    parser.add_argument('record_names',
                        type=str,
                        help='EG, a01 a02 b01 b02 x01',
                        nargs='+')
    return parser.parse_args(argv)


def wfdb2dict(input_record: str) -> dict:
    """ Estimate r-times.

    Args:
        input_record: Specification of record in wfdb database,
             eg, 'data_dir/a01'

    Return: Dict with key 'ecg' mapped to numpy array

    """
    # Read the record and extract numpy data
    record = wfdb.rdrecord(input_record)
    n_samples = record.sig_len
    assert record.p_signal.shape == (n_samples, 1)
    signal = record.p_signal[:, 0]

    assert record.fs == sample_frequency
    assert record.units == sample_units, f'record.units={record.units} sample_units={sample_units}'

    d_t = 1 / sample_frequency
    times = numpy.arange(0.0, n_samples / sample_frequency, d_t)
    return {'ecg': signal, 'times': times, 'd_t': d_t}


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    for name in args.record_names:
        _dict = wfdb2dict(os.path.join(args.wfdb_dir, name))
        with open(os.path.join(args.ecg_dir, name), mode='wb') as _file:
            pickle.dump(_dict, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
