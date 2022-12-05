from __future__ import annotations  # Enables, eg, (self: Pass1Item,

import sys
import os
import glob
import typing

import numpy

import hmm.base


class Common:

    def __init__(self, root):
        """Parameters for the apnea application.

        Args:
            root: Path to root of the hmmds project

        Essentially a namespace that functions like a FORTRAN common block.
        """

        self.data = os.path.join(root, 'derived_data/apnea')
        self.heart_rate_directory = os.path.join(self.data,
                                                 'low_pass_heart_rate')
        self.respiration_directory = os.path.join(self.data, 'respiration')
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

        self.all_names = (os.listdir(self.respiration_directory))
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
         array with shape (ntimes,3) and array[i,0] = time in minutes,
         array[i,1] = unfiltered heart rate, array[i,2] = filtered
         heart rate

    Here is the relevant code in hmmds/code/applications/apnea/rr2hr.py

    HR = rfft(hrL,131072) # 131072 is 18.2 Hrs at 2HZ
    HR[0:100] *=0 # Drop frequencies below (100*60)/65536=0.09/min
    HR[4000:] *=0 # and above (4000*60)/65536=3.66/min
    hrL = irfft(HR)

    """
    with open(path, 'r') as data_file:
        data = [[float(x) for x in line.split()] for line in data_file]
    return numpy.array(data)


def read_respiration(path: str) -> numpy.ndarray:
    """Args:
        path: File to read

    Returns:
         array with shape (ntimes,4) and array[i,0] = time in minutes,
         array[i,1:4] = Respiration vector (?Fisher linear discriminant?)

    The relevant code is hmmds/code/applications/apnea/respire.py

    """
    with open(path, 'r') as data_file:
        data = [[float(x) for x in line.split()] for line in data_file]
    return numpy.array(data)


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
    heart_rate_path = os.path.join(common.heart_rate_directory, name)
    respiration_path = os.path.join(common.respiration_directory, name)
    raw_h = read_respiration(heart_rate_path)
    raw_r = read_respiration(respiration_path)

    # Ensure that measurement times are the same.  ToDo: Why are
    # there more heart_rate data points?
    n_r = len(raw_r)
    n_h = len(raw_h)
    if t_max is None:
        limit = min(n_r, n_h)
    else:
        limit = min(n_r, n_h, t_max)

    time_difference = raw_r[:limit, 0] - raw_h[:limit, 0]
    assert numpy.abs(time_difference).max() == 0.0

    return {
        'respiration_data':
            raw_r[:limit, 1:],  # Don't store time data
        'filtered_heart_rate_data':
            raw_h[:limit, -1]  # Store only filtered heart rate
    }


def pattern_heart_rate_respiration_data(patterns: list, common: Common) -> list:
    """Prepare a list of data for names specified by patterns

    Args:
        patterns: Eg, ['b','c']
        common: Instance of Common that holds paths and parameters

    Returns:
        A list of dicts suitable as data for observaton.FilteredHeartRate_Respiration

    """

    return_list = []
    for letter in patterns:
        paths = glob.glob('{0}/{1}*'.format(common.heart_rate_directory,
                                            letter))
        for name in (os.path.basename(path) for path in paths):
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
      data:   A list of R times (peak of ecg in a heartbeat)
      w:      window size.  Look backwards and forwards at w R-times.

    Return: heart rate deviations
    
    Calculate a list of heart rate deviations sampled at 2 HZ.  The
    diviation is the jitter interpolated between the R time before the
    sample and the R time after the sample.  The jitter is the
    fraction of a pulse period by which an actual R time differs from
    the expected R time (the average time of the time before and the
    time after the beat in question).

    """

    # Create a list of jitters
    jitter = []
    for i in range(w, len(data) - w):
        sum = 0.0
        for d in (list(range(-w, 0)) + list(range(1, w + 1))):
            sum += data[i + d]
        T = sum / (2 * w)  # Expected time for data[i] if uniform
        P = (data[i + w] - data[i - w]) / (2 * w)  # Avg pulse period
        j = (data[i] - T) / P  # Fraction of pulse period by which
        # data[i] is early or late
        j = max(min(0.25, j), -0.25)  # Clip to +/- 0.25
        jitter.append([data[i], j])
    # Create an array of heart rate deviations that is uniformly
    # sampled at 2 HZ
    TF = jitter[-1][0]
    L = int(2 * TF)
    hrd = numpy.zeros(L)
    t_old = 0
    j_old = 1.0
    t_new = jitter[0][0]
    j_new = jitter[0][1]
    i = 0
    for k in range(L):
        t = k / 2.0
        if t > t_new:
            i += 1
            if i >= len(jitter):
                break
            t_old = t_new
            j_old = j_new
            t_new = jitter[i][0]
            j_new = jitter[i][1]
        djdt = (j_new - j_old) / (t_new - t_old)
        hrd[k] = j_old + (t - t_old) * djdt

    return hrd


if __name__ == "__main__":
    rv = read_expert('../../../raw_data/apnea/summary_of_training', 'a05')
    samples_per_minute = 10
    print(rv[13:15])
    print(rv[13:15].repeat(samples_per_minute))
    sys.exit(0)
    #sys.exit(main())
