from __future__ import annotations  # Enables, eg, (self: Pass1Item,

import sys
import os
import typing
import pickle
import argparse

import numpy
import scipy.signal

import hmm.base


def common_arguments(parser: argparse.ArgumentParser):
    """Common arguments to add to parsers

    Args:
        parser: Created elsewhere by argparse.ArgumentParser

    Add common arguments for apnea processing.  Make these arguments
    so that they can be modified from command lines during development
    and testing.

    """
    parser.add_argument('--root',
                        type=str,
                        default='../../../../../',
                        help='parent directory of src and build')
    parser.add_argument('--derived_apnea_data',
                        type=str,
                        default='build/derived_data/apnea',
                        help='path from root to derived apnea data')
    # Group that are relative to derived_apna
    parser.add_argument('--heart_rate_dir',
                        type=str,
                        default='Lphr',
                        help='path from derived_apnea to heart rate dir')
    parser.add_argument('--respiration_dir',
                        type=str,
                        default='Respire',
                        help='path from derived_apnea to respiration dir')
    parser.add_argument('--peak_scale',
                        type=float,
                        default=0.7,
                        help='Threshold for detecting ECG peaks')
    parser.add_argument('--pass1',
                        type=str,
                        default='pass1_report',
                        help='path from derived_apnea to the file')
    parser.add_argument('--models_dir',
                        type=str,
                        default='models',
                        help='path from derived_apnea to models dir')
    #
    parser.add_argument('--rtimes',
                        type=str,
                        default='raw_data/Rtimes',
                        help='path from root to files created by wfdb')
    parser.add_argument('--expert',
                        type=str,
                        default='raw_data/apnea/summary_of_training',
                        help='path from root to expert annotations')
    for file_name, arg_name in zip(
            'model_A4 model_C2 model_Low model_Medium model_High'.split(),
            'Amodel   BCmodel  modelLow  modelMedium  modelHigh'.split()):
        parser.add_argument(f'--{arg_name}', type=str, default=file_name)
    parser.add_argument('--iterations',
                        type=int,
                        default=20,
                        help='Training iterations')
    parser.add_argument('--low_line',
                        type=float,
                        default=1.82,
                        help='Boundary for pass1 classification')
    parser.add_argument('--high_line',
                        type=float,
                        default=2.60,
                        help='Boundary for pass1 classification')
    parser.add_argument('--stat_slope',
                        type=float,
                        default=0.5,
                        help='FixMe for pass1 classification')


def join_common(args: argparse.Namespace):
    """ Process common arguments

    Args:
        args: Namespace that includes common arguments

    Join partial paths specified as defaults or on a command line.

    """

    # Add derived_data prefix to paths in that directory
    args.derived_apnea_data = os.path.join(args.root, args.derived_apnea_data)
    for name in 'heart_rate_dir respiration_dir pass1 models_dir'.split():
        setattr(args, name,
                os.path.join(args.derived_apnea_data, getattr(args, name)))

    # Add models_dir prefix to paths in that directory
    for name in 'Amodel   BCmodel  modelLow  modelMedium  modelHigh'.split():
        setattr(args, name, os.path.join(args.models_dir, getattr(args, name)))

    args.rtimes = os.path.join(args.root, args.rtimes)
    args.expert = os.path.join(args.root, args.expert)

    args.a_names = [f'a{i:02d}' for i in range(1, 21)]
    args.b_names = [f'b{i:02d}' for i in range(1, 5)]
    args.c_names = [f'c{i:02d}' for i in range(1, 11)]
    args.x_names = [f'x{i:02d}' for i in range(1, 36)]
    args.all_names = args.a_names + args.b_names + args.c_names + args.x_names


def parse_args(argv):
    """ Example for reference and testing
    """

    parser = argparse.ArgumentParser(description='Do not use this code')
    ##### Testing ######
    common_arguments(parser)
    parser.add_argument('--sample_rate_in',
                        type=int,
                        default=2,
                        help='Samples per second of input')
    parser.add_argument('--sample_rate_out',
                        type=int,
                        default=10,
                        help='Samples per minute for results')
    parser.add_argument('input', type=str, help='Path for reading')
    parser.add_argument('output', type=str, help='Path for writing')
    args = parser.parse_args(argv)
    ##### Testing ######
    join_common(args)
    return args

def read_train_log(path: str) -> numpy.ndarray:
    """Read a text file created by train.py

    Args:
        path: Path to log file

    """

    def parse_line(line):
        result = {}
        parts = line.split()
        for i, key in enumerate(parts):
            if (key[0] == 'L' and len(key) > 1) or key in 'prior U/n'.split():
                result[key] = float(parts[i + 1])
        return result

    with open(path, 'r') as log_file:
        lines = log_file.readlines()
    column_dict = {key: [value] for key, value in parse_line(lines[0]).items()}
    for line in lines[1:]:
        _dict = parse_line(line)
        for key, value in _dict.items():
            column_dict[key].append(value)
    for key, value in column_dict.items():
        column_dict[key] = numpy.array(value)
    return column_dict


def read_ecgs(args):
    ecgs = []
    for name in args.records:
        path = os.path.join(args.rtimes, name + '.ecg')
        with open(path, 'rb') as _file:
            ecgs.append(pickle.load(_file)['raw'])
    if not args.tag_ecg:
        return ecgs

    result = []
    n_before, n_after, n_slow = args.before_after_slow
    tags = numpy.arange(2 + n_before + n_after, dtype=int)
    for ecg in ecgs:
        class_ = numpy.zeros(len(ecg), dtype=int)
        peaks, _ = scipy.signal.find_peaks(ecg / args.peak_scale,
                                           height=1.0,
                                           distance=40)
        last_stop = 0
        for peak in peaks:
            start = peak - n_before
            stop = peak + n_after + 2
            # Don't tag segments that overlap each other or the ends of
            # the data.
            if start >= last_stop and stop <= len(ecg):
                class_[start:stop] = tags
                last_stop = stop
    result.append(hmm.base.JointSegment({"class": class_, "ecg": ecg}))
    return result


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    for key, value in args.__dict__.items():
        print(f'{key}: {value}')

    print(f"{args.root=} {args.rtimes=}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
