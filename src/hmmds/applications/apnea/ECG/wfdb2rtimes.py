""" wfdb2rtimes.py: Use wfdb code to read wfdb records from PhysioNet
and write rtimes in text format.

python wfdb2rtimes.py --detector elgendi a03 wfdbdir a03_elgendi
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
import ecgdetectors


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Calculate rtimes from ecg')
    parser.add_argument('--detector',
                        type=str,
                        default='elgendi',
                        help='''Choose from:
    elgendi, matched, kalidas engzee, christov, hamilton pan_tompkins, wqrs''')
    parser.add_argument('record_name', type=str, help='eg, a03')
    parser.add_argument('wfdb_dir', type=str, help='Path to ecg data')
    parser.add_argument('result', type=str, help='Path to rtimes data')
    return parser.parse_args(argv)


def qrs(record_id: str, detector,
        sample_frequency) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    """ Estimate r-times.

    Args:
        input_record: Eg, 'wfdb_dir/a03'
        detector: 

    Return: Array of times in seconds

    """
    # Read the record and extract numpy data
    record = wfdb.rdrecord(record_id)
    n_samples = record.sig_len
    assert record.p_signal.shape == (n_samples, 1)
    signal = record.p_signal[:, 0]
    assert sample_frequency == record.fs

    indices = numpy.array(detector(signal))
    r_times = indices / sample_frequency
    return r_times, indices


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    sample_frequency = 100  # Hz
    name2index = dict((name, index) for index, name in enumerate('''
Elgendi
Matched
Kalidas
Engzee
Christov
Hamilton
Pan
WQRS
'''.split()))

    detectors = ecgdetectors.Detectors(sample_frequency)
    description, detector = detectors.get_detector_list()[name2index[
        args.detector]]
    assert description.split()[0] == args.detector
    rtimes, indices = qrs(os.path.join(args.wfdb_dir, args.record_name),
                          detector, sample_frequency)
    with open(args.result, mode='w', encoding='utf-8') as _file:
        print(
            f'Indices and Rtimes/seconds. From {args.detector}(ecg). fs={sample_frequency} Hz. n_times={len(rtimes)}',
            file=_file)
        for index, time in zip(indices, rtimes):
            print(f'{index} {time}', file=_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
