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


class Pass1Item:
    """Essentially a namespace.  ToDo: Replace with dict so that reader
    needn't chase here to understand.

    Args:
        name: eg, a01
        llr: log likelihood ratio per time step; apnea/normal
        r: Ratio of high peaks to average peaks
        stat: (r + slope * llr) used to choose level
        level: Low, Medium or Low.  Model to use for classifying each minute

    """

    def __init__(self: Pass1Item, name: str, llr: float, r: float, stat: float,
                 level: str):
        self.name = name
        self.llr = llr
        self.r = r
        self.stat = stat
        self.level = level


def read_low_pass_heart_rate(path: str) -> numpy.ndarray:
    """Args:
        path: File to read

    Returns:
         (times, low_pass_hr) Times in pint seconds. Hr in pint 1/minute

    """
    with open(path, 'rb') as _file:
        _dict = pickle.load(_file)
    hr = _dict['hr_low_pass']
    #hr = _dict['hr']  # FixMe: Change name of function
    times = (numpy.arange(len(hr)) / _dict['sample_frequency']).to('seconds')
    return times, hr


def read_respiration(path: str) -> numpy.ndarray:
    """Args:
        path: File to read

    Returns:
         (times, components)  Times are in pint seconds.  Components is a numpy array
    

    """
    with open(path, 'rb') as _file:
        _dict = pickle.load(_file)
    return _dict['times'], _dict['components']


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


def heart_rate_respiration_data(name: str, args) -> dict:
    """
    Args:
        name: Eg, 'a01'
        args: holds paths and parameters
        t_max: Optional end of time series

    Returns:
        A single dict (not a list of dicts)

    t_max enables truncation to the length of expert markings
    """
    heart_rate_path = os.path.join(args.heart_rate_dir, name + '.lphr')
    respiration_path = os.path.join(args.respiration_dir, name + '.resp')
    h_times, raw_h = read_low_pass_heart_rate(heart_rate_path)
    r_times, raw_r = read_respiration(respiration_path)

    # heart rate is sampled at 2 Hz and respiration is 10 per minute
    h_to_r = numpy.searchsorted(
        r_times.to('seconds').magnitude,
        h_times.to('seconds').magnitude)
    # raw_h[t] and raw_r[h_to_r[t]] refer to data at about the same time

    assert r_times[-1] <= h_times[
        -1], 'Respiration sample after last heart rate sample'
    i_max = numpy.searchsorted(
        h_times.to('seconds').magnitude, r_times[-1].to('seconds').magnitude)
    resampled_r = raw_r[h_to_r[numpy.arange(i_max)]]

    # Assert that arrays to return have same length
    assert len(resampled_r) == len(raw_h[:i_max])
    return {
        'respiration_data': resampled_r,
        'filtered_heart_rate_data': raw_h[:i_max].to('1/minutes').magnitude,
        'times': h_times
    }


def list_heart_rate_respiration_data(names: list, args) -> list:
    """Prepare a list of data for names specified by patterns

    Args:
        names: Eg, ['a01 a02 a03'.split()]
        common: Instance of Common that holds paths and parameters

    Returns:
        A list of dicts suitable as data for observaton.FilteredHeartRate_Respiration

    """

    return_list = []
    for name in names:
        return_list.append(heart_rate_respiration_data(name, args))
    return return_list


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


# FixMe: bundles are gone
def heart_rate_respiration_bundle_data(name: str,
                                       args) -> hmm.base.Bundle_segment:

    samples_per_minute = 10
    tags = read_expert(args.expert, name).repeat(samples_per_minute)
    underlying = heart_rate_respiration_data(name, common)
    len_respiration = len(underlying['respiration_data'])
    len_hr = len(underlying['filtered_heart_rate_data'])
    n_times = min(len(tags), len_respiration, len_hr)
    underlying = heart_rate_respiration_data(name, common, t_max=n_times)
    return hmm.base.Bundle_segment(tags[:n_times], underlying)


def rtimes2dev(data, n_ecg, w=1):
    """ Create heart rate deviations with uniform sample time.
    
    Args:
      data:   A numpy array of R times (peak of ecg in a heartbeat)
      w:      window size.  Look backwards and forwards at w R-times.

    Return: heart rate deviations
    
    Calculate a list of heart rate deviations sampled at 2 HZ.  The
    diviation is the jitter interpolated between the R time before the
    sample and the R time after the sample.  The jitter is the
    fraction of a pulse period by which an actual R time differs from
    the expected R time (the average time of the time before and the
    time after the beat in question).

    """

    # jitters is an array of deviations of rtime from prediction
    jitter = numpy.zeros(len(data))
    for i in range(w, len(data) - w):
        # Find expected time for data[i] if rtime intervals are uniform
        t_hat = (data[i - w:i].sum() + data[i + 1:i + w + 1].sum()) / (2 * w)
        d_t_hat = (data[i + w] - data[i - w]) / (2 * w)  # Avg pulse period
        # Fraction of pulse period by which data[i] is early or late
        fraction = (data[i] - t_hat) / d_t_hat
        # Clip to +/- 0.25
        jitter[i] = max(min(0.25, fraction), -0.25)
    # Create an array of heart rate deviations that is uniformly
    # sampled at 2 HZ
    t_final = n_ecg // 100  # in seconds.  Ecg sampled at 100 Hz
    length = t_final * 2  # Output sampled at 2 Hz
    times = numpy.arange(length) / 2.0  # Times in seconds of result
    start, stop = numpy.searchsorted(times, [data[0], data[-1]])
    result = numpy.empty(len(times))
    result[start:stop] = jitter[numpy.searchsorted(data, times[start:stop])]
    result[:start] = result[start]
    result[stop:] = result[stop - 1]
    return result


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    for key, value in args.__dict__.items():
        print(f'{key}: {value}')

    print(f"{args.root=} {args.rtimes=}")
    # FixMe: bundles are gone
    bundle = read_masked_ecg('a01', args)
    print(f"{len(bundle)=}")
    print(f"{bundle[0:5].bundles=}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
