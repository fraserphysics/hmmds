from __future__ import annotations  # Enables, eg, (self: Pass1Item,

import sys
import os
import glob

import numpy

import hmm.base


def common_args(parser):
    """ Add arguments required by many functions that operate on the apnea data
    """
    for path_name in 'heart_rate respiration models data expert'.split():
        parser.add_argument('--' + path_name,
                            type=str,
                            help='path to directory')
    parser.add_argument('--Amodel',
                        type=str,
                        help='Model of a-files for first classification pass')
    parser.add_argument(
        '--BCmodel',
        type=str,
        help='Model of b and c files for first classification pass')
    parser.add_argument('--iterations',
                        type=int,
                        help='Training iterations',
                        default=20)
    parser.add_argument('--pass1',
                        type=str,
                        help='path to result of pass1',
                        default='pass1_report')
    parser.add_argument('--low_line',
                        type=float,
                        help='For first pass classification')
    parser.add_argument('--high_line',
                        type=float,
                        help='For first pass classification')
    parser.add_argument('--stat_slope',
                        type=float,
                        help='For first pass classification')


class Pass1Item:
    """ Essentially a namespace

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

    Here is the relevant code in hmmds3/code/applications/apnea/rr2hr.py

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

    The relevant code is hmmds3/code/applications/apnea/respire.py

    """
    with open(path, 'r') as data_file:
        data = [[float(x) for x in line.split()] for line in data_file]
    return numpy.array(data)


def read_expert(path: str, name: str) -> numpy.array:
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
        parts = data_file.readline().split()
        while len(parts) == 0 or parts[0] != name:
            parts = data_file.readline().split()

        hour = 0
        marks = []
        # Read lines like: "8 AAAAAAAAA"
        parts = data_file.readline().split()
        while len(parts) == 2:
            assert hour == int(parts[0])
            marks += parts[1]
            parts = data_file.readline().split()
            hour += 1
    # Translate letters N,A to 0,1 and return numpy array
    return numpy.array([mark_dict[mark] for mark in marks], numpy.int32)


#TODO: use this function in test.py
def heart_rate_respiration_data(heart_rate_path: str,
                                respiration_path: str,
                                t_max=None) -> dict:
    """
    Args:
        heart_rate_path: path to a single file
        respiration_path: path to a single file
        t_max: Optional end of time series

    Returns:
        A single dict (not a list of dicts)

    t_max enables truncation to the length of expert markings
    """
    raw_h = read_respiration(heart_rate_path)
    raw_r = read_respiration(respiration_path)

    # Ensure that measurement times are the same.  ToDo: Why are
    # there more heart_rate data points?
    n_r = len(raw_r)
    n_h = len(raw_h)
    limit = min(n_r, n_h)
    if t_max is not None:
        assert t_max <= limit
        limit = t_max

    time_difference = raw_r[:limit, 0] - raw_h[:limit, 0]
    assert numpy.abs(time_difference).max() == 0.0

    return {
        'respiration_data':
            raw_r[:limit, 1:],  # Don't store time data
        'filtered_heart_rate_data':
            raw_h[:limit, -1]  # Store only filtered heart rate
    }


def pattern_heart_rate_respiration_data(args, patterns: list):
    """Prepare a list data for records specified by patterns

    Args:
        args:
        patterns: Eg, ['b','c']

    Returns:
        A list of dicts suitable as data for observaton.FilteredHeartRate_Respiration

    use glob.glob to get lists of record names and
    heart_rate_respiration_data to construct the list

    """

    return_list = []
    for letter in patterns:
        for hr_path in glob.glob('{0}/{1}*'.format(args.heart_rate, letter)):
            r_path = '{0}/{1}'.format(args.respiration,
                                      os.path.basename(hr_path))
            return_list.append(heart_rate_respiration_data(hr_path, r_path))
    return return_list


#TODO: use this function in test.py
def heart_rate_respiration_bundle_data(heart_rate_path, respiration_path,
                                       expert_path, name) -> list:

    samples_per_minute = 10
    tags = read_expert(expert_path, name).repeat(samples_per_minute)
    underlying = heart_rate_respiration_data(heart_rate_path,
                                             respiration_path,
                                             t_max=len(tags))

    return hmm.base.Bundle_segment(tags, underlying)


if __name__ == "__main__":
    rv = read_expert('../../../raw_data/apnea/summary_of_training', 'a05')
    samples_per_minute = 10
    print(rv[13:15])
    print(rv[13:15].repeat(samples_per_minute))
    sys.exit(0)
    #sys.exit(main())
