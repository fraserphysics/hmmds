"""optimize.py Use scipy optimze to improve detection performance

python optimize.py --records a01 a02 -- foo

This script evaluates classification performance of parameter vectors
by calling subprocesses to initialize, train, and apply models.

Decision variables:

# Low Pass Period seconds
LPP = 26
# Respiration Center frequency cpm
RC = 14.1
# Respiration Width cpm todo
RW = 4.58
# Filter for Respiration Smoothing in cpm.
RS = .449
# Prominence Threshold
PT = 4.8
# Exponential weight for varg component
VP = 1.0
# Exponential weight for interval component
IP = 0.032
# Detection threshold
THRESHOLD = 84

Time to run:
real	159m0.970s
user	310m48.969s
sys	4m2.288s
"""

import sys
import argparse
import subprocess
import pathlib
import pickle

import numpy
import scipy.optimize

import utilities


def parse_args(argv):
    """ A single line argument
    """

    parser = argparse.ArgumentParser("Map model name to parameter argunments")
    utilities.common_arguments(parser)
    parser.add_argument('--debug',
                        action='store_true',
                        help="Print issued commands to stdout")
    parser.add_argument('out', type=str, help='path for writing initial model')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


class Objective(dict):
    """Use call method as objective function for optimization.
    Initialization requires the exactly the following key word
    arguments:

    threshold: float apnea detection threshold
    ar: str integer Auto-Regressive order
    fs: str integer model sample frequency in samples per minute
    lpp: float Low Pass Period in seconds
    rc: float Center frequency of respiration filter in cpm
    rw: float Width of respiration filter in cpm
    rs: float Width of low pass filter for dc respiration signal
    pt: float Prominence Threshold for detecting peaks
    vp: str 1.0 Power wighting for Varg component of observations
    ip: float Power weighting for Interval component of observations
    model_dir: str path to models
    records: iterable of record names, eg, 'a01 a02 b01 c01'.split()

    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.model_dir = self.pop('model_dir')
        self.records = self.pop('records')
        assert set('threshold ar fs lpp rc rw rs pt vp ip'.split()) == set(
            self.keys())
        self.model_keys = 'ar fs lpp rc rw rs pt vp ip'.split()
        self.variable_names = 'threshold lpp rc rw rs pt ip'.split()

    def make_model(self):
        """Make a trained HMM with parameters self[key] for key in
        self.model_keys.  The file name encodes parameter values and
        the Makefile defines how to make and train the model.

        """
        model_list = ['varg4state_']
        for key in self.model_keys:
            value = self[key]
            if key in self.variable_names:
                model_list.append(f'{key}{value}')
            else:
                model_list.append(f'{key}{value}')
        model_list.append('_masked')
        model_path = pathlib.Path(self.model_dir, ''.join(model_list))
        subprocess.run(f'make {model_path}'.split(), check=True)
        return model_path

    def make_config(self):
        """Make a config file with prominence threshold self.pt

        """
        path = f'norm_config{self["pt"]}.pkl'
        subprocess.run(('make', path), check=True)
        return path

    def set(self, vector):
        """Assign variable components to self
        """
        assert len(vector) == len(self.variable_names)
        for value, key in zip(vector, self.variable_names):
            self[key] = value

    def __call__(self, vector):
        self.set(vector)
        config_path = self.make_config()
        model_path = self.make_model()
        counts = numpy.zeros(4)
        for record_name in self.records:
            model_record = utilities.ModelRecord(model_path, record_name)
            model_record.classify(self['threshold'])
            counts += model_record.score()
        pathlib.Path(config_path).unlink()
        pathlib.Path(model_path).unlink()
        return counts[1] + counts[2]


def interpret(path):
    """
    """
    with open(path, 'rb') as _file:
        result, cost_initial, cost_final = pickle.load(_file)
    
    objective = Objective(
        threshold=84.0,
        ar='16',
        fs='5',
        lpp=28.9,
        rc=14.1,
        rw=4.58,
        rs=.449,
        pt=4.8,
        vp='1.0',
        ip=0.032,
        model_dir='',
        records='',
    )

    objective.set(result.x)
    print(f'{cost_initial=} {cost_final=}')
    for key, value in objective.items():
        print(f'{key}: {value}')


def main(argv=None):
    """ Optimize parameters for classifying apnea
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    objective = Objective(
        threshold=84.0,
        ar='16',
        fs='5',
        lpp=28.9,
        rc=14.1,
        rw=4.58,
        rs=.449,
        pt=4.8,
        vp='1.0',
        ip=0.032,
        model_dir=args.model_dir,
        records=args.records,
    )
    initial = numpy.array([objective[key] for key in objective.variable_names])
    cost_initial = objective(initial)
    print(f'{cost_initial=}')
    return 0
    #                        'threshold lpp   rc    rw   rs   pt  ip'
    lower_bounds = numpy.array([1.0e-4, 15.0, 12.0, 1.0, 0.2, 1, 0.01])
    upper_bounds = numpy.array([1.0e4, 60.0, 16.0, 8.0, 1.0, 8, 100])
    # Fail on a07 with pt=12.742645786248001 no peaks.  Ok with
    # pt8.257354213751997
    bounds = scipy.optimize.Bounds(lower_bounds,
                                   upper_bounds,
                                   keep_feasible=True)
    options = dict(ftol=.01, xtol=.1)
    result = scipy.optimize.minimize(objective,
                                     initial,
                                     method='powell',
                                     bounds=bounds)
    cost_final = objective(result.x)
    with open(args.out, 'wb') as _file:
        pickle.dump((result, cost_initial, cost_final), _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
