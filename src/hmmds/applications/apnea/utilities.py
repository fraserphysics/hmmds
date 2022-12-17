from __future__ import annotations  # Enables, eg, (self: Pass1Item,

import sys
import os
import glob
import typing
import pickle

import numpy

import hmm.base


class Common:

    def __init__(self, root):
        """Parameters for the apnea application.

        Args:
            root: Path to root of the hmmds project

        Essentially a namespace that functions like a FORTRAN common
        block.  It would be better to put this information in default
        command line arguments and argparse.  """

        self.data = os.path.join(root, 'build/derived_data/apnea')
        self.heart_rate_directory = os.path.join(self.data,
                                                 'Lphr')
        self.respiration_directory = os.path.join(self.data, 'Respire')
        self.expert = os.path.join(root, 'raw_data', 'apnea',
                                   'summary_of_training')
        self.pass1 = os.path.join(self.data, 'pass1_report')
        self.models = os.path.join(self.data, 'models')
        self.Amodel = os.path.join(self.models, 'model_A2')
        self.BCmodel = os.path.join(self.models, 'model_C1')
        self.modelLow = os.path.join(self.models, 'model_Low')
        self.modelMedium = os.path.join(self.models, 'model_Medium')
        self.modelHigh = os.path.join(self.models, 'model_High')

        self.iterations = 20
        self.low_line = 1.82
        self.high_line = 2.60
        self.stat_slope = 0.5

        # 'a01' is a name
        self.all_names = [
            os.path.splitext(os.path.basename(path))[0] for path in
            glob.glob(os.path.join(self.respiration_directory, '*.resp'))]
        self.a_names = list(filter(lambda name: name[0] == 'a', self.all_names))
        self.b_names = list(filter(lambda name: name[0] == 'b', self.all_names))
        self.c_names = list(filter(lambda name: name[0] == 'c', self.all_names))
        self.x_names = list(filter(lambda name: name[0] == 'x', self.all_names))

    def get(self, key):
        return getattr(self, key)


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
    times = (numpy.arange(len(hr))/_dict['sample_frequency']).to('seconds')
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


def heart_rate_respiration_data(name: str, common: Common, t_max=None) -> dict:
    """
    Args:
        name: Eg, 'a01'
        common: instance of Common that holds paths and parameters
        t_max: Optional end of time series

    Returns:
        A single dict (not a list of dicts)

    t_max enables truncation to the length of expert markings
    """
    heart_rate_path = os.path.join(common.heart_rate_directory, name+'.lphr')
    respiration_path = os.path.join(common.respiration_directory, name+'.resp')
    h_times, raw_h = read_low_pass_heart_rate(heart_rate_path)
    r_times, raw_r = read_respiration(respiration_path)

    # heart rate is sampled at 2 Hz and respiration is 10 per minute
    h_to_r = numpy.searchsorted(r_times.to('seconds').magnitude, h_times.to('seconds').magnitude)
    # raw_h[t] and raw_r[h_to_r[t]] refer to data at about the same time

    assert r_times[-1] <= h_times[-1],'Respiration sample after last heart rate sample'
    i_max = numpy.searchsorted(h_times.to('seconds').magnitude, r_times[-1].to('seconds').magnitude)
    resampled_r = raw_r[h_to_r[numpy.arange(i_max)]]

    # Assert that arrays to return have same length
    assert len(resampled_r) == len(raw_h[:i_max])  
    return {
        'respiration_data': resampled_r,
        'filtered_heart_rate_data': raw_h[:i_max].to('1/minutes').magnitude,
        'times':h_times
    }


def list_heart_rate_respiration_data(names: list, common: Common) -> list:
    """Prepare a list of data for names specified by patterns

    Args:
        names: Eg, ['a01 a02 a03'.split()]
        common: Instance of Common that holds paths and parameters

    Returns:
        A list of dicts suitable as data for observaton.FilteredHeartRate_Respiration

    """

    return_list = []
    for name in names:
        return_list.append(heart_rate_respiration_data(name, common))
    return return_list


def heart_rate_respiration_bundle_data(
        name: str, common: Common) -> hmm.base.Bundle_segment:

    samples_per_minute = 10
    tags = read_expert(common.expert, name).repeat(samples_per_minute)
    underlying = heart_rate_respiration_data(name, common)
    len_respiration = len(underlying['respiration_data'])
    len_hr = len(underlying['filtered_heart_rate_data'])
    n_times = min(len(tags), len_respiration, len_hr)
    underlying = heart_rate_respiration_data(name, common, t_max=n_times)
    return hmm.base.Bundle_segment(tags[:n_times], underlying)


def rtimes2dev(data, w=1):
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
    t_initial = data[0]
    delta_t = data[-1] - t_initial
    # How many samples at 2 Hz fit strictly inside the interval
    length = int(2 * delta_t - 1e-6)
    remainder = delta_t - length / 2.0
    times = numpy.arange(length) / 2.0 + t_initial + remainder / 2.0
    assert remainder > 0
    # Using 10 assumes heart rate is less than 10*2/sec = 1200 bpm
    assert data[0] < times[0] < data[10]
    assert data[-10] < times[-1] < data[-1]
    return jitter[numpy.searchsorted(data, times)]


if __name__ == "__main__":
    """Test/exercise the code in this module.
    """
    root = '../../../../'
    common = Common(root)
    print(f'b_names={common.get("b_names")}')
    expert_path = os.path.join(root, *'raw_data apnea summary_of_training'.split())
    annotations = read_expert(expert_path, 'a05')
    samples_per_minute = 10
    print(annotations[13:15])
    print(annotations[13:15].repeat(samples_per_minute))
    sys.exit(0)
    #sys.exit(main())
