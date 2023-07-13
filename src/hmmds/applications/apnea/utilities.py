from __future__ import annotations  # Enables, eg, (self: Pass1Item,

import sys
import os
import typing
import pickle
import argparse

import numpy
import numpy.fft
import scipy.signal
import pint

import hmm.base

PINT = pint.UnitRegistry()


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
                        default='../../../../',
                        help='parent directory of src and build')
    parser.add_argument('--derived_apnea_data',
                        type=str,
                        default='build/derived_data/apnea',
                        help='path from root to derived apnea data')
    parser.add_argument('--records',
                        type=str,
                        nargs='+',
                        help='eg, --records a01 x02 -- ')
    # Group that are relative to derived_apna
    parser.add_argument('--rtimes',
                        type=str,
                        default='raw_data/Rtimes',
                        help='path from root to files created by wfdb')
    parser.add_argument('--expert',
                        type=str,
                        default='raw_data/apnea/summary_of_training',
                        help='path from root to expert annotations')
    parser.add_argument('--iterations',
                        type=int,
                        default=20,
                        help='Training iterations')
    parser.add_argument('--heart_rate_sample_frequency',
                        type=int,
                        default=40,
                        help='In samples per minute')
    parser.add_argument(
        '--trim_start',
        type=int,
        default=0,
        help='Number of minutes to drop from the beginning of each record')


def join_common(args: argparse.Namespace):
    """ Process common arguments

    Args:
        args: Namespace that includes common arguments

    Join partial paths specified as defaults or on a command line.

    """

    # Add derived_data prefix to paths in that directory
    args.derived_apnea_data = os.path.join(args.root, args.derived_apnea_data)

    args.heart_rate_sample_frequency *= PINT('1/minutes')
    args.trim_start *= PINT('minutes')

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


def read_expert(path: str, name: str) -> numpy.ndarray:
    """ Create int array for record specified by name.
    Args:
        path: Location of expert annotations file
        name: Record to report, eg, 'a01'

    Returns:
        array with array[t] = 0 for normal, and array[t] = 1 for apnea

    """
    mark_dict = {'N': 0, 'A': 1}
    with open(path, 'r') as data_file:

        # Skip to line that starts with name
        line = data_file.readline()
        if len(line) == 0:
            raise RuntimeError(f'{path} has no lines')
        parts = line.split()
        while len(parts) == 0 or parts[0] != name:
            line = data_file.readline()
            if len(line) == 0:
                raise RuntimeError(f'{path} has no line for {name}')
            parts = line.split()

        hour = 0
        marks: typing.List[str] = []
        # Read lines like: "8 AAAAAAAAA"
        parts = data_file.readline().split()
        while len(parts) == 2:
            assert hour == int(parts[0])
            marks += parts[1]
            parts = data_file.readline().split()
            hour += 1
    # Translate letters N,A to 0,1 and return numpy array
    return numpy.array([mark_dict[mark] for mark in marks], numpy.int32)


def window(F: numpy.ndarray,
           t_sample,
           center,
           width,
           shift=False) -> numpy.ndarray:
    """ Multiply F by a Gaussian window

    Args:
        F: The RFFT of a time series f
        t_sample: The time between samples in f
        center: The center frequency in radians per unit
        width: Sigma in radians per unit
        shift: Shift phase pi/2
    """
    # FixMe: Is this right?
    omega_max = numpy.pi / t_sample
    n_center = len(F) * (center / omega_max).to('').magnitude
    n_width = len(F) * (width / omega_max).to('').magnitude
    delta_n = numpy.arange(len(F)) - n_center
    denominator = (2 * n_width * n_width)
    assert denominator > 0
    result = F * numpy.exp(-(delta_n * delta_n) / denominator)
    if shift:
        return result * 1j
    return result


def filter_hr(raw_hr: numpy.ndarray,
              sample_period: float,
              low_pass_width,
              bandpass_center,
              skip=1,
              custom=None) -> dict:
    """ Calculate filtered heart rate
 
    Args:
        raw_hr:
        sample_period:
        low_pass_width:
        bandpass_center:  To capture 14 cycle per minute respiration

    Return: {'slow': x, 'fast': y, 'respiration':z}
    """

    n = len(raw_hr)
    HR = numpy.fft.rfft(raw_hr, 131072)
    low_pass = numpy.fft.irfft(
        window(HR, sample_period, 0 / sample_period, low_pass_width))
    BP = window(HR, sample_period, bandpass_center, low_pass_width)
    band_pass = numpy.fft.irfft(BP)
    SBP = window(HR, sample_period, bandpass_center, low_pass_width, shift=True)
    shift = numpy.fft.irfft(SBP)
    TEMP = numpy.fft.rfft(numpy.sqrt(shift * shift + band_pass * band_pass),
                          131072)
    respiration = numpy.fft.irfft(
        window(TEMP, sample_period, 0 / sample_period, low_pass_width / 2))

    result = {
        'slow': low_pass[:n:skip],
        'fast': band_pass[:n:skip],
        'respiration': respiration[:n:skip]
    }
    if not isinstance(custom, tuple):
        return result
    C = window(HR, sample_period, custom[0], custom[1])
    c = numpy.fft.irfft(C)
    result[custom[2]] = (C, c)
    return result


def read_slow_fast_respiration(args, name='a03'):
    """Read heart rate and return three filtered versions
    """

    path = os.path.join(args.derived_apnea_data,
                        f'../ECG/{name}_self_AR3/heart_rate')
    with open(path, 'rb') as _file:
        _dict = pickle.load(_file)
    f_in = _dict['sample_frequency'].to('1/minute').magnitude
    f_out = args.heart_rate_sample_frequency.to('1/minute').magnitude
    trim_samples = int(
        (args.heart_rate_sample_frequency * args.trim_start).to(''))
    skip = int(f_in / f_out)
    assert f_in == f_out * skip, f'{f_in=} {f_out=} {skip=}'
    raw_hr = _dict['hr'].to('1/minute').magnitude
    result = filter_hr(raw_hr,
                       0.5 * PINT('seconds'),
                       low_pass_width=2 * numpy.pi / (15 * PINT('seconds')),
                       bandpass_center=2 * numpy.pi * 14 / PINT('minutes'),
                       skip=skip)
    result['trim_samples'] = trim_samples
    result['sample_frequency'] = f_out * PINT('1/minute')
    return result


def read_slow_respiration(args, name='a03'):
    input_ = read_slow_fast_respiration(args, name)
    result = {}
    for key in 'slow respiration'.split():
        trim_samples = input_['trim_samples']
        if trim_samples == 0:
            result[key] = input_[key]
        else:
            result[key] = input_[key][trim_samples:-trim_samples]
    return result


def read_slow_respiration_class(args, name='a03'):
    """Add class to dict from read_slow_respiration
    """

    f_s_float = args.heart_rate_sample_frequency.to('1/minute').magnitude
    samples_per_minute = int(f_s_float)
    assert f_s_float - samples_per_minute == 0.0, f'Conversion error: {f_s_float=} {samples_per_minute=}'
    raw_dict = read_slow_respiration(args, name)
    path = os.path.join(args.root, 'raw_data/apnea/summary_of_training')
    raw_dict['class'] = read_expert(path, name).repeat(samples_per_minute)

    length = min(*[len(x) for x in raw_dict.values()])
    for key, value in raw_dict.items():
        raw_dict[key] = value[:length]
    return raw_dict


def read_slow(args, name='a03'):
    input_ = read_slow_fast_respiration(args, name)
    trim_samples = input_['trim_samples']
    if trim_samples == 0:
        return {'slow': input_['slow']}
    return {'slow': input_['slow'][trim_samples:-trim_samples]}


def read_slow_class(args, name='a03'):
    """Add class to dict from read_slow
    """

    f_s_float = args.heart_rate_sample_frequency.to('1/minute').magnitude
    samples_per_minute = int(f_s_float)
    assert f_s_float - samples_per_minute == 0.0, f'Conversion error: {f_s_float=} {samples_per_minute=}'
    raw_dict = read_slow(args, name)
    path = os.path.join(args.root, 'raw_data/apnea/summary_of_training')
    raw_dict['class'] = read_expert(path, name).repeat(samples_per_minute)
    length = min(*[len(x) for x in raw_dict.values()])
    for key, value in raw_dict.items():
        raw_dict[key] = value[:length]
    return raw_dict


# I put this in utilities enable apnea_train.py to run.
class State:
    """For defining HMM graph

    Args:
        successors: List of names (dict keys) of successor states
        probabilities: List of float probabilities for successors
        class_index: Integer class
        trainable: List of True/False for transitions described above
        prior: Optional parameters for observation model
    """

    def __init__(self,
                 successors,
                 probabilities,
                 class_index,
                 trainable=None,
                 prior=None):
        self.successors = successors
        self.probabilities = probabilities
        self.class_index = class_index
        # Each class_index must be an int because the model will be a
        # subclass of hmm.base.IntegerObservation
        if trainable:
            self.trainable = trainable
        else:
            self.trainable = [True] * len(successors)
        self.prior = prior

    def __str__(self):
        result = [f'{self.__class__} instance\n']
        result.append(f'class: {self.class_index}, prior: {self.prior}\n')
        result.append(
            f'{"successor":15s} {"probability":11s} {"trainable":9s}\n')
        for successor, probability, trainable in zip(self.successors,
                                                     self.probabilities,
                                                     self.trainable):
            result.append(
                f'{successor:15s} {probability:11.3g} {trainable:9b}\n')
        return ''.join(result)


def print_chain_model(slow, weight, key2index):
    """Print information to understand heart rate model performance.

    Args:
        slow: An AutoRegressive observation model from hmm.C or hmm.observe_float
        weight: An array of weights for each state
        key2index: Maps state keys to state indices for hmm
    """
    print(
        f'\nindex {"name":14s} {"weight":9s} {"variance":9s} {"a/b":6s} {"alpha":9s}'
    )
    for key, index in key2index.items():
        if key[-1] == '0' or key in 'N_noise normal_switch A_noise apnea_switch'.split(
        ):
            print(
                f'{index:3d}   {key:14s} {weight[index]:<9.3g} {slow.variance[index]:9.3g} {slow.beta[index]/slow.alpha[index]:6.1f} {slow.alpha[index]:9.2e}'
            )


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
