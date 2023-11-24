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
    # Group that are relative to derived_apnea
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
                        default=10,
                        help='Training iterations')
    parser.add_argument('--model_sample_frequency',
                        type=int,
                        default=6,
                        help='In samples per minute')
    parser.add_argument('--heart_rate_sample_frequency',
                        type=int,
                        default=24,
                        help='In samples per minute')
    parser.add_argument('--AR_order',
                        type=int,
                        help="Number of previous values for prediction.")
    parser.add_argument(
        '--power_and_threshold',
        type=float,
        nargs=2,
        default=(1.6, 1.0),
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
        default=8.0,
        help='Period in seconds of low pass filter for heart rate')
    parser.add_argument(
        '--band_pass_center',
        type=float,
        default=16.0,
        help='Frequency in cycles per minute for heart rate -> respiration')
    parser.add_argument(
        '--band_pass_width',
        type=float,
        default=4.0,
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
    args.model_sample_frequency *= PINT('1/minutes')
    args.trim_start *= PINT('minutes')
    args.low_pass_period *= PINT('seconds')
    args.band_pass_center /= PINT('minutes')
    args.band_pass_width /= PINT('minutes')

    args.a_names = [f'a{i:02d}' for i in range(1, 21)]
    args.b_names = [f'b{i:02d}' for i in range(1, 5)]  #b05 is no good
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


class HeartRate:
    """ Reads, holds and manipulates the derived heart rate for a record.

    Args:
        args: From parse_args
        record_name: EG 'a01'

    Many methods set values of instance attributes.

    """

    # UPPERCASE Variables are in frequency space
    # pylint: disable=attribute-defined-outside-init, invalid-name
    def __init__(self: HeartRate,
                 args,
                 record_name: str,
                 config,
                 normalize=False):

        self.args = args
        self.record_name = record_name
        self.config = config

        # Read the raw heart rate
        path = args.heart_rate_path_format.format(record_name)
        with open(path, 'rb') as _file:
            _dict = pickle.load(_file)
        assert set(_dict.keys()) == set('hr sample_frequency'.split())
        assert _dict['sample_frequency'].to('Hz').magnitude == 2

        self.hr_sample_frequency = _dict['sample_frequency']
        self.raw_hr = _dict['hr'].to('1/minute').magnitude

        if normalize:
            self.raw_hr *= config.norm_avg / Pass1(record_name,
                                                   args).statistic_2()

        # Set up for fft based filtering
        self.fft_length = 131072
        self.RAW_HR = numpy.fft.rfft(self.raw_hr, self.fft_length)
        self.hr_omega_max = numpy.pi * self.hr_sample_frequency

        # Attach attributes of args to self
        self.model_sample_frequency = args.model_sample_frequency
        self.hr_2_model_decimate = self._calculate_skip(
            1 / self.model_sample_frequency)
        self.n_model = (len(self.raw_hr) - 1) // self.hr_2_model_decimate + 1

    def dict(self: HeartRate, items, item_args=None):
        """ Return a dict of specified items sampled at rate for model

        Args:
            items: An iterable of keys
            item_args: Dict that may have arguments for get methods

        Return: dict

        """
        result = {}
        for name in items:
            if item_args and name in item_args:
                result[name] = getattr(self, f'get_{name}')(**item_args[name])
            else:
                result[name] = getattr(self, f'get_{name}')()

        lengths = numpy.zeros(len(items), dtype=int)
        for index, name in enumerate(items):
            if item_args and name in item_args and 'pad' in item_args[name]:
                lengths[index] = len(result[name]) - item_args[name]['pad']
            else:
                lengths[index] = len(result[name])

        for name, value in result.items():
            if item_args and name in item_args and 'pad' in item_args[name]:
                result[name] = value[:lengths.min() + item_args[name]['pad']]
            else:
                result[name] = value[:lengths.min()]

        return result

    def get_raw_hr(self: HeartRate):
        """Return the raw heart rate sampled at the model sample frequency
        """
        return self.raw_hr[::self.hr_2_model_decimate]

    def read_expert(self: HeartRate):
        """Read expert annotations, assign to self and return as an array
        """
        path = os.path.join(self.args.root,
                            'raw_data/apnea/summary_of_training')
        self.expert = read_expert(path, self.record_name)
        return self.expert

    def get_class(self: HeartRate):
        """Return expert annotations with repetitions for
        model_sample_frequency
        """

        if not hasattr(self, 'expert'):
            raise RuntimeError('Call read_expert before calling get_class')
        repeat = self.args.model_sample_frequency.to('1/minute').magnitude
        assert repeat - int(repeat) == 0.0
        return self.expert.repeat(int(repeat))

    def filter_hr(self: HeartRate,
                  resp_pass_center=15 / PINT('minute'),
                  resp_pass_width=3 / PINT('minute'),
                  envelope_smooth=1.5 / PINT('minute'),
                  low_pass_width=7.5 / PINT('minute')):
        """Calculate filtered heart rate properties

        Args:
            resp_pass_center: Center frequency of respiration filter
            resp_pass_width: width of respiration filter
            envelope_smooth: For smoothing amplitude of respiration envelope
            low_pass_width: For smoothing the heart rate signal

        Assigns the following attributes of self.  Shapes match
        self.raw_hr:

        resp_pass: Raw heart rate filtered 12 to 18 cpm
        envelope: Positive envelope of resp_pass
        respiration: Smoothed version of envelope
        notch: Raw heart rate with 12 to 18 cpm dropped (respiration)
        slow: Low pass filtered heart rate

        """

        omega_center = resp_pass_center * 2 * numpy.pi
        omega_width = resp_pass_width * 2 * numpy.pi
        omega_envelope = envelope_smooth * 2 * numpy.pi
        omega_low_pass = low_pass_width * 2 * numpy.pi
        n_t = len(self.raw_hr)

        sample_period_in = 1 / self.hr_sample_frequency
        resp_pass = numpy.fft.irfft(
            window(self.RAW_HR, sample_period_in, omega_center, omega_width))

        # This block calculates a positive envelope of resp_pass.  FixMe:
        # I'm not sure it's right.  SBP is spectral domain of band pass
        # shifted by pi/2
        SBP = window(self.RAW_HR,
                     sample_period_in,
                     omega_center,
                     omega_width,
                     shift=True)
        shifted = numpy.fft.irfft(SBP)
        RESPIRATION = numpy.fft.rfft(
            numpy.sqrt(shifted * shifted + resp_pass * resp_pass),
            self.fft_length)
        self.envelope = numpy.fft.irfft(RESPIRATION)[:n_t]
        self.respiration = numpy.fft.irfft(
            window(RESPIRATION, sample_period_in, 0 / sample_period_in,
                   omega_envelope))[:n_t]
        self.resp_pass = resp_pass[:n_t]

        n_low, n_high = (int(
            self.fft_length *
            (x.to('Hz').magnitude / self.hr_omega_max.to('Hz').magnitude))
                         for x in (omega_center - omega_width,
                                   omega_center + omega_width))
        NOTCH = self.RAW_HR.copy()
        NOTCH[n_low:n_high] = 0.0
        self.notch = numpy.fft.irfft(NOTCH)[:n_t]
        self.slow = numpy.fft.irfft(
            window(self.RAW_HR, sample_period_in, 0 * omega_low_pass,
                   omega_low_pass))[:n_t]

    def find_peaks(
            self: HeartRate,
            distance=0.417 * PINT('minutes'),
            wlen=1.42 * PINT('minutes'),
            prominence=None,  # In beats per minute
    ):
        """Find peaks in the low pass filtered heart rate signal

        Args:
            distance: Minimum time (pint) between peaks
            wlen: Window length (time as pint quantity)
            prominance: Minimum prominence for detection

        """
        if prominence is None:
            prominence = self.config.min_prominence
        s_f_hz = self.hr_sample_frequency.to('Hz').magnitude
        distance_samples = distance.to('seconds').magnitude * s_f_hz
        wlen_samples = wlen.to('seconds').magnitude * s_f_hz

        self.peaks, properties = scipy.signal.find_peaks(
            self.slow,
            distance=distance_samples,
            prominence=prominence,
            wlen=wlen_samples)
        self.peak_prominences = properties['prominences']
        return self.peaks

    def _calculate_skip(self: HeartRate,
                        sample_period_out,
                        sample_period_in=None):
        """Calculate integer decimation factor

        """

        def seconds(time):
            return time.to('s').magnitude

        if sample_period_in is None:
            sample_period_in = 1 / self.hr_sample_frequency
        skip = int(seconds(sample_period_out) / seconds(sample_period_in))
        # Check for float -> int trouble
        assert skip * seconds(sample_period_in) == seconds(
            sample_period_out
        ), f'{skip=} {seconds(sample_period_out)=} {seconds(sample_period_in)=}'
        return skip

    def get_resp_pass(self: HeartRate):
        """Return heart rate signal filtered to pass respiration
        frequencies.  Decimate to model_sample_frequency.

        """
        if not hasattr(self, 'resp_pass'):
            raise RuntimeError('Call filter_hr before calling get_resp_pass')
        result = self.resp_pass[::self.hr_2_model_decimate]
        assert result.shape == (self.n_model,)
        return result

    def get_notch(self: HeartRate):
        """Return heart rate signal filtered to block respiration
        frequencies.  Decimate to model_sample_frequency.

        """
        if not hasattr(self, 'notch'):
            raise RuntimeError('Call hr_2_respiration before calling get_notch')
        result = self.notch[::self.hr_2_model_decimate]
        assert result.shape == (self.n_model,)
        return result

    def get_envelope(self: HeartRate):
        """Return envelope of heart rate signal filtered to pass
        respiration frequencies.  Decimate to model_sample_frequency.

        """
        if not hasattr(self, 'envelope'):
            raise RuntimeError('Call filter_hr before calling get_envelope')
        result = self.envelope[::self.hr_2_model_decimate]
        assert result.shape == (self.n_model,)
        return result

    def get_slow(self: HeartRate, pad=0):
        """Return low pass filtered heart rate signal.  Decimate to
        model_sample_frequency.

        """
        if not hasattr(self, 'slow'):
            raise RuntimeError('Call filter_hr before calling get_slow')
        result = self.slow[::self.hr_2_model_decimate]
        assert result.shape == (self.n_model,)
        if pad == 0:
            return result
        padded = numpy.empty(len(result) + pad)
        padded[:pad] = result[0]
        padded[pad:] = result
        return padded

    def get_hr_respiration(self: HeartRate, pad=0):
        """Return a time series of 2d vectors: result[t] =
        (hr,respiration)

        """
        hr = self.get_slow()
        respiration = self.get_envelope()
        assert hr.shape == respiration.shape == (self.n_model,)
        result = numpy.stack((hr, respiration), axis=1)
        assert result.shape == (self.n_model, 2)
        if pad == 0:
            return result
        padded = numpy.empty((len(result) + pad, 2))
        padded[:pad] = result[0]
        padded[pad:] = result
        return padded

    def get_peak(self: HeartRate):
        """Return array with ones where there are peaks and zeros
        elesewhere.  Use model_sample_frequency.

        """

        if not hasattr(self, 'peaks'):
            raise RuntimeError('Call find_peaks before calling get_peak')
        # There could be a peak after self.n_model *
        # self.hr_2_model_decimate
        peak = numpy.zeros(self.n_model + 1, dtype=numpy.int32)
        locations = self.peaks // self.hr_2_model_decimate
        peak[locations] = 1
        return peak[:self.n_model]

    def get_interval(self: HeartRate):
        """Return an interval duration signal sampled at
        model_sample_frequency.

        """
        if not hasattr(self, 'peaks'):
            raise RuntimeError('Call find_peaks before calling get_interval')
        interval = numpy.ones(self.n_model, dtype=float) * -1
        for t_start, t_stop in zip(self.peaks[:-1], self.peaks[1:]):
            i_start, i_stop = (
                t // self.hr_2_model_decimate for t in (t_start, t_stop))
            interval[i_start:i_stop] = t_stop - t_start
        first_i, last_i = (t // self.hr_2_model_decimate
                           for t in (self.peaks[0], self.peaks[-1]))
        # The loop assigned interval from first_i up to (not including) last_i
        interval[:first_i] = interval[first_i]
        interval[last_i:] = interval[last_i - 1]
        assert interval.min() > -1
        return interval


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
    with open(path, encoding='utf-8', mode='r') as data_file:

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
        t_sample: The time (pint) between samples in f
        center: The center frequency in radians per pint time
        width: Sigma in radians per pint time
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
    notch=(12 * PINT('1/minutes'), 18 * PINT('1/minutes')),
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
    n_low, n_high, n_top = (int(
        len(HR) * (2 * numpy.pi * x / omega_max).to('Hz').magnitude)
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


def calculate_skip(sample_period_in, sample_period_out):
    """Calculate integer decimation factor

    """

    def seconds(time):
        return time.to('s').magnitude

    skip = int(seconds(sample_period_out) / seconds(sample_period_in))
    # Check for float -> int trouble
    assert skip * seconds(sample_period_in) == seconds(
        sample_period_out
    ), f'{skip=} {seconds(sample_period_out)=} {seconds(sample_period_in)=}'
    return skip


def hr_2_respiration(
    raw_hr: numpy.ndarray,
    sample_period_in,
    sample_period_out=PINT('minute') / 40,
    bandpass_center=15 / PINT('minute'),
    bandpass_width=3 / PINT('minute'),
    smooth_width=1.5 / PINT('minute')
) -> dict:
    """Calculate filtered heart rate
 
    Args:
        raw_hr: Array of estimated hear rates
        sample_period: pint quantity. Time between raw_hr samples
        bandpass_center: Center frequency of respiration filter
        bandpass_width: width of respiration filter
        smooth_width: For smoothing amplitude of envelope

    Return: {'filtered': y, 'envelope': z, 'times':t}

    The number and times of samples in y and z match the input raw_hr.

    """

    omega_center = bandpass_center * 2 * numpy.pi
    omega_width = bandpass_width * 2 * numpy.pi
    omega_smooth = smooth_width * 2 * numpy.pi

    n = len(raw_hr)
    HR = numpy.fft.rfft(raw_hr, 131072)
    BP = window(HR, sample_period_in, omega_center, omega_width)
    bandpass = numpy.fft.irfft(BP)

    # This block calculates a positive envelope of bandpass.  FixMe:
    # I'm not sure it's right.  SBP is spectral domain of band pass
    # shifted by pi/2
    SBP = window(HR, sample_period_in, omega_center, omega_width, shift=True)
    shifted = numpy.fft.irfft(SBP)
    RESPIRATION = numpy.fft.rfft(
        numpy.sqrt(shifted * shifted + bandpass * bandpass), 131072)
    envelope = numpy.fft.irfft(RESPIRATION)
    respiration = numpy.fft.irfft(
        window(RESPIRATION, sample_period_in, 0 / sample_period_in,
               omega_smooth))

    skip = calculate_skip(sample_period_in, sample_period_out)
    return {
        'fast': bandpass[:n:skip],
        'respiration': respiration[:n:skip],
        'times': numpy.arange(n // skip) * sample_period_out
    }


def read_hr(args, record_name):
    """Verify format of pickled heart rate and return as array

    Args:
        args: Provides args.heart_rate_path_format
        record_name: EG, 'a01'

    Return: (raw_hr, sample_frequency)

    raw_hr is a numpy array with ~8 hours at 2Hz, ~57,600 samples
    sample_frequency is 2 Hz as a pint quantity
    """
    path = args.heart_rate_path_format.format(record_name)
    with open(path, 'rb') as _file:
        _dict = pickle.load(_file)
    assert set(_dict.keys()) == set('hr sample_frequency'.split())
    sample_frequency = _dict['sample_frequency']
    assert sample_frequency.to('Hz').magnitude == 2
    raw_hr = _dict['hr'].to('1/minute').magnitude

    return raw_hr, sample_frequency


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

    def __init__(self: State,
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

    def set_transitions(self: State, successors, probabilities, trainable=None):
        if trainable is None:
            self.trainable = [True] * len(successors)
        else:
            self.trainable = trainable
        self.successors = successors
        self.probabilities = probabilities

    def __str__(self: State):
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

    def __init__(self: IntervalObservation, density_functions: tuple, args,
                 power):
        r"""Output of state is a float.  The density is 1.0 if the
        state is an apnea state, and if the state is normal the
        density is given by a density ratio."

        Args:
            density_functions: density_functions[s] is a callable: float -> float
            config: Parameters for density_functions 
            power: Exponent of likelihood, for weighting wrt other components of joint observation.

        """
        self.n_states = len(density_functions)
        self.density_functions = density_functions
        self.config = args.config
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
                self._y.reshape(-1, 1), self.config)).reshape(-1)**self.power
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
        hmm.base.JointSegment(read_slow_peak_interval(model.args, record_name))
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

    def statistic_4(self: Pass1, low=12, high=18) -> float:
        """Spectral power in respiration bands.  Range in cpm

        """
        # argmax finds first place inequality is true
        channel_low = numpy.argmax(self.frequencies > low)
        channel_high = numpy.argmax(self.frequencies > high)

        return self.psd[channel_low:channel_high].sum()


class ModelRecord:

    def __init__(self: ModelRecord, model_path: str, record_name: str):
        """Holds a model and observations for a single record

        Args:
            model: Path to pickled HMM
            record: The name, eg, 'a01' of the record

        Note: Subsequent processing uses the "args" attribute of the
        pickled model.  A ModelRecord instance only uses information
        from the caller that are in the arguments to __init__, namely
        model_path and record_name.  In particular no other
        information from the command line arguments of the caller have
        any effects.

        """
        self.record_name = record_name
        with open(model_path, 'rb') as _file:
            self.model = pickle.load(_file)
        self.samples_per_minute = int(
            self.model.args.model_sample_frequency.to('1/minute').magnitude)
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
            threshold:  Higher value -> less (Normal -> Apnea) errors
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


def peaks_intervals(args, record_names):
    """Calculate (prominence, period) pairs.

    Args:
        args: command line arguments
        record_names: Specifies data to analyze

    The returned intervals are in minutes
    """

    # Calculate (prominence, period) pairs
    peak_dict = {0: [], 1: []}
    for record_name in record_names:
        heart_rate = HeartRate(args, record_name, args.config)
        heart_rate.filter_hr()
        heart_rate.read_expert()
        slow = heart_rate.slow
        _class = heart_rate.expert

        sample_frequency = heart_rate.hr_sample_frequency
        sample_frequency_cpm = int(sample_frequency.to('1/minute').magnitude)
        if 'config' in args:
            min_prominence = args.config.min_prominence
        else:  # When called by config_stats which makes config
            min_prominence = args.min_prominence

        peak_times, properties = peaks(slow, sample_frequency, min_prominence)
        for index in range(len(peak_times) - 1):
            t_peak = peak_times[index]
            prominence_t = properties['prominences'][index]
            period_t = (peak_times[index + 1] - t_peak) / sample_frequency_cpm
            temp = t_peak // sample_frequency_cpm
            if temp >= len(_class):
                break
            peak_dict[_class[temp]].append((prominence_t, period_t))
    return peak_dict


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

    def drop_big(interval):
        """Return x_i after dropping values larger than limit
        """
        return interval[numpy.nonzero(interval < limit)]

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


def apnea_pdf(x, _):
    return numpy.ones(x.shape)


def normal_pdf(x_in, config):
    x = config.pdf_ratio.x
    y = config.pdf_ratio.y
    big = numpy.nonzero(x_in > x[-1])
    small = numpy.nonzero(x_in < x[0])
    result = config.normal_pdf_spline(x_in)
    result[big] = y[-1]
    result[small] = y[0]
    return result


def make_interval_pdfs(args, records=None):
    """Extrapolate result of estimating the pdf ratio to range [0, \infty)

    """

    if records is None:
        records = args.a_names
    # Find peaks
    peak_dict = peaks_intervals(args, records)

    limit = 2.2  # No intervals longer than this for pdf ratio fit
    sigma = 0.1  # Kernel width
    _lambda = 0.06  # Regularize pdf ratio fit
    pdf_ratio = make_density_ratio(peak_dict, limit, sigma, _lambda)

    return pdf_ratio


def read_lphr_respiration_class(args, record_name):
    """FixMe: put this in model_init.py
    """

    keys = 'hr_respiration class'.split()
    item_args = {'hr_respiration': {'pad': args.AR_order}}

    # develop.HMM.read_y_with_class calls this with self.args, and
    # apnea_train.main wraps the result in hmm.base.JointSegment

    assert args.config.normalize == args.normalize

    hr_instance = HeartRate(args, record_name, args.config, args.normalize)
    hr_instance.read_expert()
    hr_instance.filter_hr()

    return hr_instance.dict(keys, item_args)


def read_lphr_respiration(args, record_name):
    """FixMe: put this in model_init.py
    """

    keys = ['hr_respiration']
    item_args = {'hr_respiration': {'pad': args.AR_order}}

    # develop.HMM.read_y_with_class calls this with self.args, and
    # apnea_train.main wraps the result in hmm.base.JointSegment

    assert args.config.normalize == args.normalize

    hr_instance = HeartRate(args, record_name, args.config, args.normalize)
    hr_instance.filter_hr()

    return hr_instance.dict(keys, item_args)


def read_slow_class_peak_interval(args, record_name):
    """FixMe: put this in model_init.py Called by HMM, and returns a
    dict of observation components

    Args:
        args: From HMM.args
        record_name: EG, 'a01'

    Components are slow, peak, interval and class.

    """
    keys = 'slow peak interval class'.split()
    item_args = {'slow': {'pad': args.AR_order}}

    # develop.HMM.read_y_with_class calls this with self.args, and
    # apnea_train.main wraps the result in hmm.base.JointSegment

    assert args.config.normalize == args.normalize

    hr_instance = HeartRate(args, record_name, args.config, args.normalize)
    hr_instance.read_expert()
    hr_instance.filter_hr()
    hr_instance.find_peaks()

    return hr_instance.dict(keys, item_args)


def read_slow_peak_interval(args, record_name):
    """FixMe: put this in model_init.py Called by HMM, and returns a
    dict of observation components

    Args:
        args: From HMM.args
        record_name: EG, 'a01'

    Components are slow, peak, and interval.

    """
    keys = 'slow peak interval'.split()
    item_args = {'slow': {'pad': args.AR_order}}

    # develop.HMM.read_y_with_class calls this with self.args, and
    # apnea_train.main wraps the result in hmm.base.JointSegment

    assert args.config.normalize == args.normalize

    hr_instance = HeartRate(args, record_name, args.config, args.normalize)
    hr_instance.filter_hr()
    hr_instance.find_peaks()

    return hr_instance.dict(keys, item_args)


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
