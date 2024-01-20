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
        self.variable_values = {}
        for key, value in self.items():
            if key in self.variable_names:
                assert len(value) == 2
                self.variable_values[key] = value[0]
                continue
            assert key in self.model_keys
            assert isinstance(value, str)

    def make_model(self):
        """Make a trained HMM with parameters self[key] for key in
        self.model_keys.  The file name encodes parameter values and
        the Makefile defines how to make and train the model.

        """
        model_list = ['varg4state_']
        for key in self.model_keys:
            if key in self.variable_names:
                model_list.append(f'{key}{self.variable_values[key]}')
            else:
                model_list.append(f'{key}{self[key]}')
        model_list.append('_masked')
        model_path = pathlib.Path(self.model_dir, ''.join(model_list))
        subprocess.run(f'make {model_path}'.split(), check=True)
        return model_path

    def make_config(self):
        """Make a config file with prominence threshold self.pt

        """
        path = f'norm_config{self.variable_values["pt"]}.pkl'
        subprocess.run(('make', path), check=True)
        return path

    def set(self, vector):
        """Map vector to self.variable_values
        """
        assert len(vector) == len(self.variable_names)
        for z, key in zip(vector, self.variable_names):
            self.variable_values[key] = self[key][0] + z * self[key][1]

    def __call__(self, vector):
        self.set(vector)
        config_path = self.make_config()
        model_path = self.make_model()
        counts = numpy.zeros(4)
        for record_name in self.records:
            model_record = utilities.ModelRecord(model_path, record_name)
            model_record.classify(self.variable_values['threshold'])
            counts += model_record.score()
        pathlib.Path(config_path).unlink()
        pathlib.Path(model_path).unlink()
        return counts[1] + counts[2]

    def interpret(self, vector):
        """Print contents of self and the parameters that vector maps
        to.

        """
        self.set(vector)
        for key, value in self.items():
            if key in self.variable_names:
                print(f'{key:8s}: {self.variable_values[key]:14.8f} <- {value}')
            else:
                print(f'{key:8s}: {value}')


def interpret(path):
    """Print information about result.  For example from ipython:

    In [1]: from optimize import Objective
    In [2]: import optimize
    In [3]: optimize.interpret('v4s_opt')

    """
    with open(path, 'rb') as _file:
        result, objective, cost_initial, cost_final = pickle.load(_file)

    print(f'{cost_initial=} {cost_final=}\n{result=}\n')
    objective.interpret(result.x)


def main(argv=None):
    """ Optimize parameters for classifying apnea
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    # (a,b) "a" is the nominal value, and "b" is roughly the deviation
    # to move from 3000 to 3500 errors
    objective = Objective(
        threshold=(2427.0, 700.0),
        ar='16',
        fs='5',
        lpp=(28.9, 25.0),
        rc=(14.1, 3.0),
        rw=(4.58, 1.5),
        rs=(.449, .02),
        pt=(4.8, 2.5),
        vp='1.0',
        ip=(0.045, 0.4),
        model_dir=args.model_dir,
        records=args.records,
    )
    dim_x = len(objective.variable_names)
    initial = numpy.zeros(dim_x)
    cost_initial = objective(initial)
    print(f'{cost_initial=}')

    # For Nelder-Mead initial_simplex centered at zero with scale
    # bigger then bumps
    initial_simplex = numpy.ones((dim_x + 1, dim_x)) / dim_x
    for i in range(dim_x):
        initial_simplex[i + 1, i] -= 1.1
    initial_simplex *= .2

    lower_bounds = -numpy.ones(dim_x) * 2.0
    upper_bounds = numpy.ones(dim_x) * 2.0
    # Fail on a07 with pt=12.742645786248001 no peaks.  Ok with
    # pt8.257354213751997.  BFGS can't handle bounds
    bounds = scipy.optimize.Bounds(lower_bounds,
                                   upper_bounds,
                                   keep_feasible=True)

    # Options are for CG or BFGS methods

    # f(x_opt) \approx 3000
    # Hope to get within 10 with dx .1 so
    gtol = 10 / .1
    # f is bumpy with dx=.1 so use
    # finite_diff_rel_step = 0.2
    eps = 0.2
    options = dict(gtol=gtol, norm=2, eps=eps)

    nm_options = dict(inital_simplex=initial_simplex, xatol=0.001, fatol=5)
    result = scipy.optimize.minimize(objective,
                                     numpy.ones(dim_x),
                                     method='Nelder-Mead',
                                     options=nm_options)
    cost_final = objective(result.x)
    with open(args.out, 'wb') as _file:
        pickle.dump((result, objective, cost_initial, cost_final), _file)

    # print a summary of the result
    interpret(args.out)

    # Method Result
    # BFGS   success: False
    # CG     success: False.  'Desired error not necessarily achieved due to precision loss.'
    # Powell success: True, but cost(result.x) > cost(initial)

    return 0


if __name__ == "__main__":
    sys.exit(main())
