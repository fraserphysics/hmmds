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
import pygraphviz

import hmm.base
import hmm.C

PINT = pint.get_application_registry()


def common_arguments(parser: argparse.ArgumentParser):
    """Common arguments to add to parsers

    Args:
        parser: Created elsewhere by argparse.ArgumentParser

    Add common arguments for apnea processing.  Make these arguments
    so that they can be modified from command lines during development
    and testing.

    """
    parser.add_argument('--records',
                        type=str,
                        nargs='+',
                        help='eg, --records a01 x02 -- ')
    # Paths
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
    parser.add_argument('--expert',
                        type=str,
                        default='raw_data/apnea/summary_of_training',
                        help='path from root to expert annotations')
    # Parameters
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
        '--threshold',
        type=float,
        default=1.0,
        help='Apnea detection threshold.  A positive in (,\infty)')
    parser.add_argument(
        '--power_dict',
        type=str,
        nargs='+',
        help='Observation components weighted by likelihood**power')
    parser.add_argument('--components',
                        type=str,
                        nargs='+',
                        help='Names of observation components')
    parser.add_argument(
        '--trim_start',
        type=int,
        default=0,
        help='Number of minutes to drop from the beginning of each record')
    parser.add_argument(
        '--low_pass_period',
        type=float,
        default=8.0,
        help='Period in seconds of low pass filter for heart rate')
    parser.add_argument(
        '--band_pass_center',
        type=float,
        default=16.0,
        help='Frequency in cycles per minute for heart rate -> resp_pass')
    parser.add_argument(
        '--band_pass_width',
        type=float,
        default=4.0,
        help='Frequency in cycles per minute for heart rate -> resp_pass')
    parser.add_argument(
        '--respiration_smooth',
        type=float,
        default=1.5,
        help=
        'Frequency in cycles per minute of smoothing for resp_pass -> respiration'
    )


def join_common(args: argparse.Namespace):
    """ Process common arguments

    Args:
        args: Namespace that includes common arguments

    Join partial paths specified as defaults or on a command line.

    """

    if args.power_dict:
        assert len(args.power_dict) % 2 == 0
        temp = dict((name, float(value)) for name, value in zip(
            args.power_dict[0::2], args.power_dict[1::2]))
        args.power_dict = temp

    # Add root prefix to paths in that directory
    args.derived_apnea_data = os.path.join(args.root, args.derived_apnea_data)
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
    args.respiration_smooth /= PINT('minutes')

    args.a_names = [f'a{i:02d}' for i in range(1, 21)]
    args.b_names = [f'b{i:02d}' for i in range(1, 5)]  #b05 is no good
    args.c_names = [f'c{i:02d}' for i in range(1, 11)]
    args.c_names.remove('c04')  # c04 has arrhythmia
    args.c_names.remove('c06')  # c06 is the same as c05
    args.x_names = [f'x{i:02d}' for i in range(1, 36)]
    args.all_names = args.a_names + args.b_names + args.c_names + args.x_names
    args.test_names = args.a_names + args.b_names + args.c_names


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


# FixMe: Not using hmm.C.HMM_SPARSE because: 1. It's not much faster
# in training for ~600 state apnea models 2. It's method forward
# crashes.  3. It doesn't have a decode method
class HMM(hmm.C.HMM):  #hmm.C.HMM or hmm.base.HMM
    """Has class_estimate method and many other variant features

    Args:
        p_state_initial : Initial distribution of states
        p_state_time_average : Time average distribution of states
        p_state2state : Probability of state given state:
            p_state2state[a, b] = Prob(s[1]=b|s[0]=a)
        y_mod : Instance of class for probabilities of observations
        args: Holds command line args and other information
        rng : Numpy generator with state

    KWArgs:
        untrainable_indices: List of ordered node pairs
        untrainable_values: Fixed transition probabilities

    Other variant features:

        * Untrainable state transition probabilities
    
        * likelihood method calculates probability of y[t]|y[:t] for
          all t

        * weights method calculates Prob (state[t]=i|y[:]) for all i
          and t.  Called by class_estimate and explore.py

    """

    def __init__(self: HMM,
                 p_state_initial: numpy.ndarray,
                 p_state_time_average: numpy.ndarray,
                 p_state2state: numpy.ndarray,
                 y_mod: hmm.simple.Observation,
                 args: argparse.Namespace,
                 rng: numpy.random.Generator,
                 untrainable_indices=None,
                 untrainable_values=None):
        """Option of holding some elements of p_state2state constant
        in reestimation.

        """
        hmm.C.HMM.__init__(self,
                           p_state_initial,
                           p_state_time_average,
                           p_state2state,
                           y_mod,
                           rng=rng)
        self.args = args
        self.untrainable_indices = untrainable_indices
        self.untrainable_values = untrainable_values

    def viz(self: HMM, dot_path=None, pdf_path=None):
        """Write graphviz representation of self to file[s]

        """

        index2key = [None] * self.n_states
        for key, index in self.args.state_key2state_index.items():
            index2key[index] = key

        n_states = len(self.p_state_initial)

        has_class = 'class' in self.y_mod

        class State:

            def __init__(state, index):
                state.index = index
                y_mod = self.y_mod
                if has_class:
                    if index in self.y_mod['class'].class2state[0]:
                        state.class_ = 'Normal'
                        state.color = 'blue'
                    else:
                        state.class_ = 'Apnea'
                        state.color = 'red'
                else:
                    state.color = 'blue'
                varg = y_mod["hr_respiration"]
                covariance = numpy.linalg.inv(varg.inverse_sigma[index])
                save = numpy.get_printoptions()['precision']
                numpy.set_printoptions(precision=3)
                analysis = numpy.linalg.eigh(covariance)
                print(f'''state {index} eigenvalues: {analysis.eigenvalues}
eigenvectors:
{analysis.eigenvectors}
''')
                Psi = varg.Psi[index]
                nu = varg.nu[index]
                state.label = f'''<<table>
<tr> <td> State {index}: {index2key[index]} </td> </tr>
<tr> <td> Prob: {self.p_state_time_average[index]:6.4f} </td> </tr>
<tr> <td> Sigma: {covariance} </td> </tr>
<tr> <td> Psi: {Psi} </td> </tr>
<tr> <td> nu: {nu} </td> </tr>
</table>>'''
                numpy.set_printoptions(precision=save)
                state.successors = []
                for state_f in range(n_states):
                    p_tran = self.p_state2state[index, state_f]
                    if p_tran <= 0.0:
                        continue
                    text = f'{p_tran:5.3g}'
                    if p_tran < 0.01:
                        text = f'{p_tran:7.1e}'
                    state.successors.append((state_f, text))

        graph = pygraphviz.AGraph(directed=True,
                                  strict=True,
                                  nodesep=1.5,
                                  ranksep=1.5)
        state_dict = {}
        for index in range(n_states):
            state_dict[index] = state = State(index)
            graph.add_node(index, shape='rectangle', label=state.label)
        if has_class:
            graph.add_subgraph(self.y_mod['class'].class2state[0],
                               cluster=True,
                               label='Normal')
            graph.add_subgraph(self.y_mod['class'].class2state[1],
                               cluster=True,
                               label='Apnea')
        for state in state_dict.values():
            for state_f, text in state.successors:
                if self.untrainable_indices and (state.index, state_f) in list(
                        zip(*self.untrainable_indices)):
                    color = 'black'
                else:
                    color = state.color
                graph.add_edge(state.index,
                               state_f,
                               color=color,
                               fontcolor=color,
                               taillabel=text,
                               labelfontsize=12)
        if dot_path:
            assert dot_path[-4:] == '.dot'
            graph.write(dot_path)
        if pdf_path:
            assert pdf_path[-4:] == '.pdf'
            graph.draw(pdf_path, prog='dot')

    # The self.args argument for the read functions was defined when
    # the HMM instance was created.  The only information propagated
    # from the caller is "record_name".
    def read_y_no_class(self: HMM, record_name):
        return self.args.read_raw_y(self.args, record_name)

    def read_y_with_class(self: HMM, record_name):
        return self.args.read_y_class(self.args, record_name)

    def reestimate(self: HMM):
        """Variant can hold some self.p_state2state values constant.

        Reestimates observation model parameters.

        """

        hmm.C.HMM.reestimate(self)
        if self.untrainable_indices is None or len(
                self.untrainable_indices) == 0:
            return
        self.p_state2state[self.untrainable_indices] = self.untrainable_values
        self.p_state2state.normalize()
        return

    def likelihood(self: HMM, y) -> numpy.ndarray:
        """Calculate p(y[t]|y[:t]) for t < len(y)

        Args:
            y: A single segment appropriate for self.y_mod.observe([y])

        Returns Prob y[t]|y[:t] for all t

        """
        self.y_mod.observe([y])
        state_likelihood = self.y_mod.calculate()
        length = len(state_likelihood)  # Less than len(y) if y_mod is
        # autoregressive
        result = numpy.empty(length)
        last = numpy.copy(self.p_state_initial)
        for t in range(length):
            last *= state_likelihood[t]
            last_sum = last.sum()  # Probability of y[t]|y[:t]
            result[t] = last_sum
            if last_sum > 0.0:
                last /= last_sum
            else:
                print(f'Zero likelihood at {t=}.  Reset.')
                last = numpy.copy(self.p_state_initial)
            self.p_state2state.step_forward(last)
        return result

    def weights(self: HMM, y) -> numpy.ndarray:
        """Calculate p(s,t|y) for t < len(y)

        Args:
            y: Data appropriate for self.y_mod.observe([y])

        """
        t_seg = self.y_mod.observe(y)
        n_times = self.y_mod.n_times
        assert t_seg[-1] == n_times

        self.alpha = numpy.empty((n_times, self.n_states))
        self.beta = numpy.empty((n_times, self.n_states))
        self.gamma_inv = numpy.empty((n_times,))
        self.state_likelihood = self.y_mod.calculate()
        assert self.state_likelihood[0, :].sum() > 0.0
        self.forward()
        self.backward()
        return self.alpha * self.beta

    def class_estimate(self: HMM,
                       y: list,
                       samples_per_minute: int,
                       threshold: float = 1.0,
                       power: dict = None) -> list:
        """ Estimate a sequence of classes
        Args:
            y: List with single element that is a time series of measurements
            samples_per_minute: Sample frequency
            threshold: >1 increase or (0:1) decrease the number of normal minutes
            power: Exponential weights of observation components

        Returns:
            Time series of class identifiers with a sample frequency 1/minute
        """
        class_model = self.y_mod['class']
        del self.y_mod['class']
        if power:
            self.y_mod.power = power
        weights = self.weights(y)
        self.y_mod['class'] = class_model  # Restore for future use

        def weights_per_minute(state_list):
            """ For each minute calculate total weight in listed states

            Args:
                state_list: States to sum over
            """
            minutes = self.y_mod.n_times // samples_per_minute
            remainder = self.y_mod.n_times % samples_per_minute
            if remainder == 0:
                # First sum is over states.  Second sum is over
                # samples in each minute.
                result = weights[:, state_list].sum(axis=1).reshape(
                    -1, samples_per_minute).sum(axis=1)
            else:
                # Both result[-2] and result[-1] are based on one
                # minute of data.  The data they depend on overlap by
                # (samples_per_minute - remainder).
                result = numpy.empty(minutes + 1)
                result[:minutes] = weights[:-remainder,
                                           state_list].sum(axis=1).reshape(
                                               -1,
                                               samples_per_minute).sum(axis=1)
                result[-1] = weights[-samples_per_minute:, state_list].sum()
            return result

        weights_normal = weights_per_minute(class_model.class2state[0])
        weights_apnea = weights_per_minute(class_model.class2state[1])
        result = weights_apnea > weights_normal * threshold
        return result

    def weights_missing(self: HMM,
                        y: list,
                        missing: str,
                        power: dict = None) -> numpy.ndarray:
        """Calculate state probabilities for observations that are missing a component
        Args:
            y: List with single element that is a time series of measurements
            missing: key for missing component of y
            power: Exponential weights of observation components

        Returns:
            Time series w[t,i] = Prob state[t] = i given y


        """
        missing_model = self.y_mod[missing]
        del self.y_mod[missing]
        if power:
            self.y_mod.power = power
        weights = self.weights(y)
        self.y_mod[missing] = missing_model  # Restore for future use
        return weights

    def estimate_missing(self: HMM,
                         y: list,
                         missing: str,
                         power: dict = None) -> list:
        """Estimate a sequence of missing data
        Args:
            y: List with single element that is a time series of measurements
            missing: key for missing component of y
            coefficients: Optional replacement for means
            power: Exponential weights of observation components

        Returns:
            Time series of estimated values of missing data

        Derived from class_estimate.  I wrote this to estimate the
        best threshold for a record.

        """
        weights = self.weights_missing(y, missing, power)

        result = []
        for weight in weights:
            result.append(numpy.dot(self.y_mod[missing].mu, weight))
        return numpy.array(result)


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
                 config=None,
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
                  respiration_smooth=1.5 / PINT('minute'),
                  low_pass_width=7.5 / PINT('minute')):
        """Calculate filtered heart rate properties

        Args:
            resp_pass_center: Center frequency of respiration filter
            resp_pass_width: width of respiration filter
            respiration_smooth: For smoothing amplitude of respiration envelope
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
        omega_envelope = respiration_smooth * 2 * numpy.pi
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
        distance_samples = int(distance.to('seconds').magnitude * s_f_hz)
        wlen_samples = int(wlen.to('seconds').magnitude * s_f_hz)

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

    def get_respiration(self: HeartRate):
        """Return respiration signal derived from heart rate signal.
        Decimate to model_sample_frequency.

        """
        if not hasattr(self, 'respiration'):
            raise RuntimeError('Call filter_hr before calling get_respiration')
        result = self.respiration[::self.hr_2_model_decimate]
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
        respiration = self.get_respiration()
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
        assert interval.min() > -1, f'{self.record_name=} {first_i=} {last_i=}'
        return interval


def read_expert(path: str, name: str) -> numpy.ndarray:
    """ Create int array for record specified by name.
    Args:
        path: Location of expert annotations file
        name: Record to report, eg, 'a01'

    Returns:
        array with array[t] = 0 for normal, and array[t] = 1 for apnea

    The sample frequency of the result is 1/minute.

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


# Putting this in utilities lets apnea_train.py run.
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
        self.has_class = 'class' in self.model.y_mod
        self.samples_per_minute = int(
            self.model.args.model_sample_frequency.to('1/minute').magnitude)
        if record_name[0] == 'x' or not self.has_class:
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
            power: Dict of exponential weights of ovservation components

        """

        if power and set(self.model.y_mod.keys()) != set(power.keys()):
            raise ValueError(f'''keys of y_mod do not match keys of power:
{self.model.y_mod.keys()=}
{power.keys()=} ''')
        self.class_from_model = self.model.class_estimate(
            self.y_raw_data, self.samples_per_minute, threshold, power)

        if self.record_name[0] == 'x':  # Used for display by explore.py
            self.class_from_expert = numpy.zeros(len(self.class_from_model),
                                                 dtype=int)

    def decode(self: ModelRecord):
        """Viterbi decode to estimate apnea or normal for each minute of data

        """

        class_model = self.model.y_mod['class']
        del self.model.y_mod['class']
        states = self.model.decode(self.y_raw_data)
        self.model.y_mod['class'] = class_model  # Restore for future use

        classes = numpy.array(
            list(state in self.model.y_mod['class'].class2state[1]
                 for state in states))
        # -(-) to get rounding up
        n_minutes = -(-len(states) // self.samples_per_minute)
        self.class_from_model = numpy.zeros(n_minutes, dtype=int)
        for minute in range(n_minutes):
            start = minute * self.samples_per_minute
            stop = min(start + self.samples_per_minute, len(states))
            if classes[start:stop].sum() > (stop - start) // 2:
                self.class_from_model[minute] = 1

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

    def best_threshold(self: ModelRecord, minimum=-3.0, maximum=3.0, levels=10):
        """Find rough approximation of threshold that minimizes error

        Args:
            minimum: Bottom of range of log_10 thresholds
            maximum: Top of range of log_10 thresholds
            levels: Number of thresholds considered

        Return: (best_threshold, class_counts)

        """
        thresholds = numpy.geomspace(10**minimum, 10**maximum, levels)
        objective_values = numpy.zeros(levels, dtype=int)
        counts = numpy.empty((levels, 4), dtype=int)
        for i, threshold in enumerate(thresholds):
            self.classify(threshold=threshold)
            counts[i, :] = self.score()
            objective_values[i] = counts[i, 1] + counts[i, 2]
        best_i = numpy.argmin(objective_values)
        return numpy.log10(thresholds[best_i]), counts[best_i]

    def cheat(self: ModelRecord):
        """ classify using expert to find best threshold

        Return: threshold
        """
        threshold, self.counts = self.best_threshold()
        self.classify(threshold)
        return threshold

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


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    for key, value in args.__dict__.items():
        print(f'{key}: {value}')

    return 0


if __name__ == "__main__":
    sys.exit(main())
