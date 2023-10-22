from __future__ import annotations  # Enables, eg, self: Pass1

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
import density_ratio

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
    parser.add_argument('--model_dir',
                        type=str,
                        default='build/derived_data/apnea/models',
                        help='path from root to hmms for heart rate')
    parser.add_argument(
        '--heart_rate_path_format',
        type=str,
        default='build/derived_data/ECG/{0}_self_AR3/heart_rate',
        help='path from root to heart rate data')
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
                        default=8,
                        help='In samples per minute')
    parser.add_argument('--AR_order',
                        type=int,
                        help="Number of previous values for prediction.")
    parser.add_argument(
        '--power_and_threshold',
        type=float,
        nargs=2,
        default=(2.0, 1.0),
        help=
        'Weight of observation component "interval" and apnea detection threshold'
    )
    parser.add_argument(
        '--trim_start',
        type=int,
        default=0,
        help='Number of minutes to drop from the beginning of each record')
    parser.add_argument(
        '--fft_width',
        type=int,
        default=4096,
        help='Number of samples for each fft for pass1 statistic')
    parser.add_argument(
        '--low_pass_period',
        type=float,
        default=15.0,
        help='Period in seconds of low pass filter for heart rate')
    parser.add_argument(
        '--band_pass_center',
        type=float,
        default=14.0,
        help='Frequency in cycles per minute for heart rate -> respiration')


def join_common(args: argparse.Namespace):
    """ Process common arguments

    Args:
        args: Namespace that includes common arguments

    Join partial paths specified as defaults or on a command line.

    """

    # Add root prefix to paths in that directory
    args.derived_apnea_data = os.path.join(args.root, args.derived_apnea_data)
    args.rtimes = os.path.join(args.root, args.rtimes)
    args.expert = os.path.join(args.root, args.expert)
    args.heart_rate_path_format = os.path.join(args.root,
                                               args.heart_rate_path_format)
    args.model_dir = os.path.join(args.root, args.model_dir)

    args.heart_rate_sample_frequency *= PINT('1/minutes')
    args.trim_start *= PINT('minutes')
    args.low_pass_period *= PINT('seconds')
    args.band_pass_center /= PINT('minutes')

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
    # *.to('Hz').magnitude enables different registries.
    omega_max = (numpy.pi / t_sample).to('Hz').magnitude
    n_center = len(F) * (center / omega_max).to('Hz').magnitude
    n_width = len(F) * (width / omega_max).to('Hz').magnitude
    delta_n = numpy.arange(len(F)) - n_center
    denominator = (2 * n_width * n_width)
    assert denominator > 0
    result = F * numpy.exp(-(delta_n * delta_n) / denominator)
    if shift:
        return result * 1j
    return result


def notch_hr(
    raw_hr: numpy.ndarray,
    sample_period=.5 * PINT('seconds'),
    notch=(50 * PINT('1/minutes'), 175 * PINT('1/minutes')),
    top=-1 * PINT('1/minutes')
) -> numpy.ndarray:
    """Calculate filtered heart rate
 
    Args:
        raw_hr:
        sample_period: Time with pint
        notch: (low, high) frequencies with pint
        top: Max frequency with pint

    Return: filtered_hr Same shape as raw_hr

    This function removes frequencies in the range defined by notch
    and frequencies above top.  The default values for notch
    correspond to the range for normal respiration.

    """

    n_t = len(raw_hr)
    HR = numpy.fft.rfft(raw_hr, 131072)

    omega_max = (numpy.pi / sample_period).to('Hz').magnitude
    n_low, n_high, n_top = (int(len(HR) * (x / omega_max).to('Hz').magnitude)
                            for x in notch + (top,))

    HR[n_low:n_high] = 0.0
    if n_top > 0:
        HR[n_top:] = 0.0

    return numpy.fft.irfft(HR)[:n_t]


def peaks(
        filtered: numpy.ndarray,  # Heart rate in beats per minute
        sample_frequency,  # A pint frequency
        prominence,  # In beats per minute
        distance=0.417 * PINT('minutes'),
        wlen=1.42 * PINT('minutes'),
):
    """Find peaks in the low pass filtered heart rate signal
    
    Args:
        filtered: Heart rate time series as array of floats
        sample_frequency: A pint frequency
        distance: Minimum time between peaks
        prominance: Minimum prominence for detection
        wlen: Window length

    Return: peaks_, properties

    peaks_ is an array of indices

    """
    s_f_hz = sample_frequency.to('Hz').magnitude
    distance_samples = distance.to('seconds').magnitude * s_f_hz
    wlen_samples = wlen.to('seconds').magnitude * s_f_hz

    peaks_, properties = scipy.signal.find_peaks(filtered,
                                                 distance=distance_samples,
                                                 prominence=prominence,
                                                 wlen=wlen_samples)
    return peaks_, properties


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
    sample_period = (1.0 / f_in) * PINT('minutes')
    trim_samples = int(
        (args.heart_rate_sample_frequency * args.trim_start).to(''))
    skip = int(f_in / f_out)
    assert f_in == f_out * skip, f'{f_in=} {f_out=} {skip=}'
    raw_hr = _dict['hr'].to('1/minute').magnitude
    # Option to normalize
    if hasattr(args, 'norm_avg'):
        divisor = Pass1(name, args).statistic_2()
        norm = divisor / args.norm_avg
        raw_hr /= norm
    # Now pad front of raw_hr to compensate for AR-order
    if args.AR_order is None:
        pad = 0
    else:
        pad = skip * args.AR_order
    padded = numpy.empty(len(raw_hr) + pad)
    padded[:pad] = raw_hr[0]
    padded[pad:] = raw_hr
    result = filter_hr(padded,
                       sample_period,
                       low_pass_width=2 * numpy.pi / args.low_pass_period,
                       bandpass_center=2 * numpy.pi * args.band_pass_center,
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

    Args:
        args:
        name:  Record name, eg, 'a03'

    Return: raw_dict

    Keys of raw_dict are 'slow' and 'class' and values are time series
    sampled at rate args.heart_rate_sample_frequency

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


def add_peaks(args, raw_dict, boundaries):
    """Add key item 'peak':values to raw_dict

    """
    slow_signal = raw_dict['slow']

    if hasattr(args, 'divisor'):
        locations, properties = peaks(slow_signal,
                                      args.heart_rate_sample_frequency,
                                      args.min_prominence / args.divisor)
        digits = numpy.digitize(properties['prominences'],
                                boundaries / args.divisor)
    else:
        locations, properties = peaks(slow_signal,
                                      args.heart_rate_sample_frequency,
                                      args.min_prominence)
        digits = numpy.digitize(properties['prominences'], boundaries)
    peak_signal = numpy.zeros(len(slow_signal), dtype=numpy.int32)
    peak_signal[locations] = digits
    raw_dict['peak'] = peak_signal
    assert peak_signal.max() > 0
    return raw_dict


def add_intervals(args, raw_dict):
    """Add key item 'interval':values to raw_dict

    """
    peak_signal = raw_dict['peak']

    interval_signal = numpy.zeros(n_times := len(peak_signal))
    locations = numpy.nonzero(peak_signal)[0]

    for t_start, t_stop in zip(locations[:-1], locations[1:]):
        interval_signal[t_start:t_stop] = t_stop - t_start
    interval_signal[:locations[0]] = locations[0]
    interval_signal[locations[-1]:] = n_times - locations[-1]

    raw_dict[
        'interval'] = interval_signal / args.heart_rate_sample_frequency.to(
            '1/minute').magnitude
    return raw_dict


def read_slow_class_peak(args, boundaries, name='a03'):
    """Add peak to dict from read_slow_class

    Args:
        args:
        boundaries:
        name:  Record name, eg, 'a03'

    Return: raw_dict

    Keys of raw_dict are 'slow', 'class', and 'peak' and values are
    time series sampled at rate args.heart_rate_sample_frequency

    """

    return add_peaks(args, read_slow_class(args, name), boundaries)


def read_slow_peak(args, boundaries, name='a03'):
    """Add peak to dict from read_slow

    Args:
        args:
        boundaries:
        name:  Record name, eg, 'a03'

    Return: raw_dict

    Keys of raw_dict are 'slow', 'class', and 'peak' and values are
    time series sampled at rate args.heart_rate_sample_frequency

    """

    return add_peaks(args, read_slow(args, name), boundaries)


def read_slow_class_peak_interval(args, boundaries, name='a03'):
    """Add peak to dict from read_slow_class

    Args:
        args:
        boundaries:
        name:  Record name, eg, 'a03'

    Return: raw_dict

    Keys of raw_dict are 'slow', 'class', and 'peak' and values are
    time series sampled at rate args.heart_rate_sample_frequency

    """

    return add_intervals(
        args, add_peaks(args, read_slow_class(args, name), boundaries))


def read_slow_peak_interval(args, boundaries, name='a03'):
    """Add peak to dict from read_slow

    Args:
        args:
        boundaries:
        name:  Record name, eg, 'a03'

    Return: raw_dict

    Keys of raw_dict are 'slow', 'class', and 'peak' and values are
    time series sampled at rate args.heart_rate_sample_frequency

    """

    return add_intervals(args, add_peaks(args, read_slow(args, name),
                                         boundaries))


def read_normalized_class(args, boundaries, name):
    args.divisor = Pass1(name, args).statistic_2()
    return read_slow_class_peak_interval(args, boundaries, name)


def read_normalized(args, boundaries, name):
    args.divisor = Pass1(name, args).statistic_2()
    return read_slow_peak_interval(args, boundaries, name)


# I put this in utilities enable apnea_train.py to run.
class State:
    """For defining HMM graph

    Args:
        successors: List of names (dict keys) of successor states
        probabilities: List of float probabilities for successors
        observation: Object used to make observation model
        trainable: List of True/False for transitions described above
        prior: Optional parameters for observation model
    """

    def __init__(self,
                 successors,
                 probabilities,
                 observation,
                 trainable=None,
                 prior=None):
        self.successors = successors
        self.probabilities = probabilities
        self.observation = observation
        if trainable:
            self.trainable = trainable
        else:
            self.trainable = [True] * len(successors)
        self.prior = prior

    def set_transitions(self, successors, probabilities, trainable=None):
        if trainable is None:
            self.trainable = [True] * len(successors)
        else:
            self.trainable = trainable
        self.successors = successors
        self.probabilities = probabilities

    def __str__(self):
        result = [f'{self.__class__} instance\n']
        result.append(f'observation: {self.observation}, prior: {self.prior}\n')
        result.append(
            f'{"successor":15s} {"probability":11s} {"trainable":9s}\n')
        for successor, probability, trainable in zip(self.successors,
                                                     self.probabilities,
                                                     self.trainable):
            result.append(
                f'{successor:15s} {probability:11.3g} {trainable:9b}\n')
        return ''.join(result)


class IntervalObservation(hmm.base.BaseObservation):
    _parameter_keys = 'density_functions n_states power'.split()

    def __init__(self: IntervalObservation, density_functions: tuple, power=5):
        r"""Output of state is a float.  The density is 1.0 if the
        state is an apnea state, and if the state is normal the
        density is given by a density ratio."

        Args:
            density_functions: density_functions[s] is a callable: float -> float
        power: Exponent of likelihood, for weighting wrt other components of joint observation.

        """
        self.n_states = len(density_functions)
        self.density_functions = density_functions
        self.power = power

    def reestimate(self: IntervalObservation, w: numpy.ndarray):
        pass

    def calculate(self: IntervalObservation) -> numpy.ndarray:
        """Return likelihoods of states given self._y, a sequence of
        classes.

        """
        self._likelihood *= 0.0
        for i_state in range(self.n_states):
            self._likelihood[:, i_state] = (self.density_functions[i_state](
                self._y.reshape(-1, 1))).reshape(-1)**self.power
        return self._likelihood


# FixMe: Test this
def interval_hmm_likelihood(model_path: str,
                            record_name: str,
                            interval_power=1.0):
    """Calculate log likelihood of model wrt data for a record
    """
    with open(model_path, 'rb') as _file:
        model = pickle.load(_file)
    del model.y_mod['class']
    model.y_mod['interval'].power = interval_power

    y_data = [
        hmm.base.JointSegment(
            read_slow_peak_interval(model.args, model.args.boundaries,
                                    record_name))
    ]
    t_seg = model.y_mod.observe(y_data)
    n_times = model.y_mod.n_times
    assert t_seg[-1] == n_times

    model.alpha = numpy.empty((n_times, model.n_states))
    model.gamma_inv = numpy.empty((n_times,))
    model.state_likelihood = model.y_mod.calculate()
    assert model.state_likelihood[0, :].sum() > 0.0
    log_likelihood = model.forward()
    return log_likelihood


class Pass1:
    """Holds statistics of a record for pass1 classification, ie, normal or apnea
    """

    def __init__(self, name, args, norm_frequency=0.3):
        """Read heart rate data and attach some analysis results to self

        Args:
            name: EG "a01"
            args:
            norm_frequency: Divide PSD by sum of channels above this frequency (in cpm)

        """

        if hasattr(args, 'model'):
            self.likelihood = interval_hmm_likelihood(args.model, name)

        self.name = name
        with open(args.heart_rate_path_format.format(name), 'rb') as _file:
            _dict = pickle.load(_file)
            sample_frequency = _dict['sample_frequency'].to(
                '1/minutes').magnitude
            hr = _dict['hr']
            trim = int(sample_frequency *
                       args.trim_start.to('minutes').magnitude)
        heart_rate = _dict['hr'].to('1/minute').magnitude[trim:]
        self.frequencies, self.psd = scipy.signal.welch(heart_rate,
                                                        fs=sample_frequency,
                                                        nperseg=args.fft_width)
        self.norm_channel = numpy.argmax(self.frequencies > norm_frequency)

    def statistic_1(self: Pass1, low=1.0, high=3.6) -> float:
        """Spectral power in the range of low frequency apnea
        oscillations.  Range in cpm

        """

        return self.statistic_3(low, high) / self.statistic_2()

    def statistic_2(self: Pass1):
        """Power in frequencies higher than norm_channel

        """
        return self.psd[self.norm_channel:].sum()

    def statistic_3(self: Pass1, low=1.0, high=3.6) -> float:
        """Spectral power in band.  Range in cpm

        """
        # argmax finds first place inequality is true
        channel_low = numpy.argmax(self.frequencies > low)
        channel_high = numpy.argmax(self.frequencies > high)

        return self.psd[channel_low:channel_high].sum()


class ModelRecord:

    def __init__(self: ModelRecord, model_path: str, record_name: str):
        """Holds a model and observations for a single record

        Args:
            model: Path to pickled HMM and its args
            record: The name, eg, 'a01' of the record
        """
        self.record_name = record_name
        with open(model_path, 'rb') as _file:
            self.model = pickle.load(_file)
        self.samples_per_minute = int(
            self.model.args.heart_rate_sample_frequency.to(
                '1/minute').magnitude)
        if record_name[0] == 'x':
            self.y_class_data = None
            self.class_from_expert = None
        else:
            self.y_class_data = [self.model.read_y_with_class(record_name)]
            self.class_from_expert = read_expert(self.model.args.expert,
                                                 self.record_name)
        self.y_raw_data = [
            hmm.base.JointSegment(self.model.read_y_no_class(record_name))
        ]
        self.class_from_model = None
        self.counts = numpy.empty(4, dtype=int)

    def classify(self: ModelRecord, threshold=None, power=None):
        """Estimate apnea or normal for each minute of data

        Args:
            threshold:  Higher value -> less Normal -> Apnea errors
            power: Exponent to adjust weight of interval component of likelihood

        """
        _power, _threshold = self.model.args.power_and_threshold
        if power is None:
            power = _power
        if threshold is None:
            threshold = _threshold

        if 'interval' in self.model.y_mod:
            self.model.y_mod['interval'].power = power

        self.class_from_model = self.model.class_estimate(
            self.y_raw_data, self.samples_per_minute, threshold)

        if self.record_name[0] == 'x':  # Used for display by explore.py
            self.class_from_expert = numpy.zeros(len(self.class_from_model),
                                                 dtype=int)

    def score(self: ModelRecord):
        """For each minute compare class estimate from model to expert

        n2n = counts[0]
        n2a = counts[1]
        a2n = counts[2]
        a2a = counts[3]
        """
        n_minutes = min(len(self.class_from_model), len(self.class_from_expert))
        self.counts *= 0
        for i in range(n_minutes):
            self.counts[2 * self.class_from_expert[i] +
                        self.class_from_model[i]] += 1
        return self.counts.copy()

    def formatted_result(self: ModelRecord,
                         report: typing.TextIO,
                         expert=False):
        """Write result to open file in format that matches expert
        """
        if expert:
            authority = self.class_from_expert
        else:
            authority = self.class_from_model
        minutes_per_hour = 60
        n_minutes = self.counts.sum()
        # -(- ...) to round up instead of down
        n_hours = -(-n_minutes // minutes_per_hour)
        print('{0}\n'.format(self.record_name), end='', file=report)
        for hour in range(n_hours):
            print(' {0:1d} '.format(hour), end='', file=report)
            minute_start = hour * minutes_per_hour
            minute_stop = min((hour + 1) * minutes_per_hour, n_minutes)
            for minute in range(minute_start, minute_stop):
                if authority[minute]:
                    print('A', end='', file=report)
                else:
                    print('N', end='', file=report)
            print('\n', end='', file=report)


def print_chain_model(y_mod, weight, key2index):
    """Print information to understand heart rate model performance.

    Args:
        y_mod: A joint observation model
        weight: An array of weights for each state
        key2index: Maps state keys to state indices for hmm
    """
    interval = y_mod['interval']
    print(f'\nindex {"name":14s} {"weight":9s}')
    for key, index in key2index.items():
        if key[-1] == '0' or key[
                -1] == '1' or key in 'N_noise N_switch A_noise A_switch'.split(
                ):
            print(f'{index:3d}   {key:14s} {weight[index]:<9.4g}')


def peaks_intervals(args, record_names, peaks_per_bin):
    """Calculate (prominence, period) pairs.  Also find boundaries for
    digitizing prominence.

    Args:
        args: command line arguments
        record_names: Specifies data to analyze
        peaks_per_bin: Number of apnea peaks between bounary levels

    The returned intervals are in minutes
    """

    f_sample = args.heart_rate_sample_frequency.to('1/minute').magnitude
    apnea_key = 1

    # Calculate (prominence, period) pairs
    peak_dict = {0: [], 1: []}
    norm_sum = 0.0
    for record_name in record_names:
        norm_sum += Pass1(record_name, args).statistic_2()
        raw_dict = read_slow_class(args, record_name)
        slow = raw_dict['slow']
        _class = raw_dict['class']
        peak_ts, properties = peaks(slow, args.heart_rate_sample_frequency,
                                    args.min_prominence)
        for index in range(len(peak_ts) - 1):
            t_peak = peak_ts[index]
            prominence_t = properties['prominences'][index]
            period_t = (peak_ts[index + 1] - t_peak) / f_sample
            class_t = _class[t_peak]
            peak_dict[class_t].append((prominence_t, period_t))

    # Calculate boundaries for prominence based on peaks during apnea
    pp_array = numpy.array(peak_dict[apnea_key]).T
    apnea_peaks = pp_array[0]
    apnea_peaks.sort()
    boundaries = []
    for index in range(0, len(apnea_peaks), peaks_per_bin):
        boundaries.append(apnea_peaks[index])
    boundaries = numpy.array(boundaries).T

    return peak_dict, boundaries, norm_sum / len(record_names)


def make_density_ratio(peak_dict, limit, sigma, _lambda):
    """  Estimate the function p_normal(x)/p_apnea(x)

    Args:
        peak_dict: Arrays of peak heights and lengths of intervals between peaks
        limit: Drop samples with length > limit
        sigma: Width of kernel function
        _lambda: Regularizes optimization

    Returns: density_ratio.DensityRatio instance

    """

    key_n = 0
    key_a = 1
    i_interval = 1

    def drop_big(x_in):
        """Return x_i after dropping values larger than limit
        """
        return x_in[numpy.nonzero(x_in < limit)]

    normal, apnea = (drop_big(numpy.array(peak_dict[key])[:,
                                                          i_interval]).reshape(
                                                              -1, 1)
                     for key in (key_n, key_a))

    # FixMe: The result is a random function.  I should study the
    # variability.
    result = density_ratio.uLSIF(normal,
                                 apnea,
                                 kernel_num=800,
                                 sigma=sigma,
                                 _lambda=_lambda)
    return result


def apnea_pdf(x):
    return numpy.ones(x.shape)


def normal_pdf(x_in, characteristics):
    x = characteristics['pdf_ratio'].x
    y = characteristics['pdf_ratio'].y
    big = numpy.nonzero(x_in > x[-1])
    small = numpy.nonzero(x_in < x[0])
    result = characteristics['normal_pdf_spline'](x_in)
    result[big] = y[-1]
    result[small] = y[0]
    return result


def make_interval_pdfs(args, records=None):
    """Extrapolate result of estimating the pdf ratio to range [0, \infty)

    """

    if records is None:
        records = args.a_names
    # Find peaks
    peaks_per_bin = 2000  # Meaninless in this context
    peak_dict, _, _ = peaks_intervals(args, records, peaks_per_bin)

    limit = 2.2  # No intervals longer than this for pdf ratio fit
    sigma = 0.1  # Kernel width
    _lambda = 0.06  # Regularize pdf ratio fit
    pdf_ratio = make_density_ratio(peak_dict, limit, sigma, _lambda)

    return pdf_ratio


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
