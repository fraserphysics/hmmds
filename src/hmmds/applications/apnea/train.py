"""train.py Command line specifies type of data and record names of data

Example:
python train.py --records a01 x02 b01 c05 --type ecg models/initial_ECG models/trained_ECG

The type selects one of the registered functions in this module.

"""
import sys
import os.path
import pickle
import argparse

import numpy

import hmm.base

import hmmds.applications.apnea.utilities
import hmmds.applications.apnea.observation
import develop
import utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Train an HMM on specified records")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('--records', type=str, nargs='+', help='EG: a01 x02')
    parser.add_argument('--type',
                        type=str,
                        help='A type registered in this module, eg, "ECG"')
    parser.add_argument('input', type=str, help='path to initial model')
    parser.add_argument('output', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


TYPES = {}  # Is populated by @register decorated functions.  The keys
# are function names, and the values are functions for reading data.


def register(func):
    """Decorator that puts function in TYPES dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    TYPES[func.__name__] = func
    return func


def segment(data):
    """Break ECG data into 15 minute segments for parallel processing

    """

    f_pm = 60 * 100  # Samples per minute
    minutes_per_segment = 15
    samples_per_segment = minutes_per_segment * f_pm

    result = []
    for n_start in range(0, len(data), samples_per_segment):
        n_stop = n_start + samples_per_segment
        result.append(data[n_start:n_stop])
    return result


@register
def segmented(args) -> list:
    """Read raw ecg data specified by args.  Sample data with 15
    minute segments to enable parallel training.

    """
    data = utilities.read_ecgs(args)
    result = []
    for data_record in data:
        result.extend(segment(data_record))
    return result


@register
def diverse(args) -> list:
    """Return a list of segment from each record specified in the args.


    one 15 minute segment Ensure that all of the data in each segment is
    plausible wrt the model in model_dir.

    """

    segment_length = 15*60*100 # 90,000 = 15 minutes, 60
                               # seconds/minute, 100 samples/second
    class RecordData:
        """Structure to hold data about a record.

        """
        def __init__(self, record_name, raw_data, model_path):
            self.name = record_name
            self.raw_data = raw_data
            self.n_data = len(raw_data)
            self.dir_path = os.path.dirname(model_path)
            states_path = os.path.join(self.dir_path, 'states', self.name)
            likelihood_path = os.path.join(self.dir_path, 'likelihood', self.name)
            with open(states_path, 'rb') as _file:
                self.states = pickle.load(_file)
            with open(likelihood_path, 'rb') as _file:
                self.likelihood = pickle.load(_file)

        def segment(self):
            """Find and return a 15 minute segment that is plausible for model

            """
            bad_spots_ = numpy.nonzero((self.states == 0) | (self.likelihood < 1.0e-70))[0]
            bad_spots = numpy.empty(len(bad_spots_)+2, dtype=int)
            bad_spots[1:-1:] = bad_spots_
            bad_spots[0] = 0
            bad_spots[-1] = self.n_data-1
            ok_lengths = bad_spots[1:] - bad_spots[:-1]
            i_start = ok_lengths.argmax()
            interval = (bad_spots[i_start]+1, bad_spots[i_start+1])
            assert self.likelihood[interval[0]:interval[1]].min() >= 1.0e-70
            interval_length = interval[1] - interval[0]
            if interval_length <= segment_length:
                assert self.name != 'a01'
                return self.raw_data[interval[0]:interval[1]]
            i_start = interval[0] + (interval_length - segment_length)//2
            return self.raw_data[i_start: i_start+segment_length]

    result = []
    records = {}
    raw_data = utilities.read_ecgs(args)
    for raw_record, name in zip(raw_data, args.records):
        records[name] = RecordData(name, raw_record, args.input)
        segment_ = records[name].segment()
        if len(segment_) < segment_length:
            print(f'Skipping record {name} because for longest segment {len(segment_)=}')
        else:
            result.append(segment_)
    return result


def compare(a, b, attribute):
    value_a, value_b = (getattr(x, attribute) for x in (a, b))
    if numpy.array_equal(value_a, value_b):
        return
    if numpy.allclose(value_a, value_b):
        print(f"values of {attribute} are close but not equal")
    else:
        print(f"values of {attribute} are not close")


def compare_hmms(a, b):
    print("Comparing two hmms")
    for attribute in "p_state_initial p_state2state p_state_time_average".split(
    ):
        compare(a, b, attribute)

    print("Comparing the underlying y models")
    y_mod_a, y_mod_b = (hmm.y_mod.underlying_model for hmm in (a, b))
    for attribute in "alpha beta coefficients norm variance".split():
        compare(y_mod_a, y_mod_b, attribute)


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.input, 'rb') as _file:
        _args, _hmm = pickle.load(_file)

    # Use the registered function to read the training data
    _args.records = args.records
    _args.input = args.input
    data = TYPES[args.type](_args)

    _hmm.multi_train(data, args.iterations)
    _hmm.strip()

    with open(args.output, 'wb') as _file:
        pickle.dump((_args, _hmm), _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
