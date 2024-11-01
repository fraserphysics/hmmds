"""toy_values.py Read computation write in form for LaTeX

"""
from __future__ import annotations  # Enables, eg, (self: LocalNonStationary

import sys
import argparse
import pickle
import os

import numpy


def parse_args(argv, arg_list):
    """Parse a command line.
    """
    parser = argparse.ArgumentParser(
        description='Translate picked values to LaTex')
    for name in arg_list:
        parser.add_argument(name, type=str, help=f'Path to pickle file')
    parser.add_argument('result', type=str, help='path for result')
    return parser.parse_args(argv)


FunctionDict = {}  # Keys are function names.  Values are functions.


def register(func):
    """Decorator that puts function in FunctionDict
    """
    FunctionDict[func.__name__] = func
    return func


@register
def data_h_view(dict_in):
    """return expressions for y_step unit_state_noise observation_noise_multiplier dt

    Args:
        dict_in: Result of running h_cli.py

    """
    t_start = dict_in['t_start']
    t_stop = dict_in['t_stop']
    t_max_dev = numpy.argmax(dict_in['y_variances'][t_start:t_stop])
    # Return some nice formats for LaTeX.  Rather than formating
    # values, check those values and use typed in strings.
    result = {
        'toyHviewSigmaEta': dict_in['unit_state_noise'][0, 0],
        'toyHviewSigmaEpsilon': dict_in['observation_noise_multiplier'][0, 0],
        'toyHviewTmaxSigma': t_max_dev,
        'toyHviewTmaxSigmaPlusOne': t_max_dev + 1,
    }
    for key in 'y_step atol dt'.split():
        result[f"toyHview{key.replace('_', '')}"] = dict_in[key]
    return result


@register
def toy_h(dict_in):
    """return expressions for intercept slope and the following args:
    n_t dev_measurement dev_state_generate dev_state_filter y_step

    toyhSigmaEta toyhSigmaEpsilon toyhDelta

    Args:
        dict_in: Result of running toy_h.py

    """
    result = {
        'toyToyhIntercept': dict_in['intercept'],
        'toyToyhSlope': dict_in['slope'],
    }

    args = dict_in['args']
    for key in 'n_t dev_measurement dev_state_generate dev_state_filter y_step'.split(
    ):
        result[f"toyToyh{key.replace('_', '')}"] = getattr(args, key)
    return result


@register
def benettin(dict_in):
    """return expressions for n_runs dev_state grid_size n_times n_runs dev_state/grid_size

    Args:
        dict_in: Result of running benettin.py

    """
    # Return nice formats for LaTeX.  Rather than formating values in
    # args, check those values and use typed in strings.
    args = dict_in['args']
    result = {}
    for key in 'n_runs dev_state grid_size n_times'.split():
        result[f"toyBenettin{key.replace('_', '')}"] = getattr(args, key)
    return result


@register
def like_lor(dict_in):
    """return expressions for n_train n_test n_quantized t_sample min_prob

    Args:
        dict_in: Result of running like_lor.py

    I'm not sure that the results are used in the LaTeX.
    """
    args = dict_in['args']
    result = {}
    for key in 'n_train n_test n_quantized t_sample min_prob'.split():
        result[f"toyLikeLor{key.replace('_', '')}"] = getattr(args, key)
    return result


def main(argv=None):
    """Study dependence of relative entropy of Kalman filters on
    actual sample time and assumed observation noise.

    """
    if argv is None:
        argv = sys.argv[1:]
    # List of targets in Rules.mk
    arg_list = 'data_h_view toy_h benettin like_lor'.split()
    args = parse_args(argv, arg_list)

    result = {}

    for name in arg_list:
        path = getattr(args, name)
        with open(path, 'rb') as file_:
            dict_in = pickle.load(file_)
        result.update(FunctionDict[name](dict_in))

    replacements = {
        1e-4: '10^{-4}',
        1e-5: '10^{-5}',
        1e-6: '10^{-6}',
        1e-7: '10^{-7}',
        1e-8: '10^{-8}',
        1e-9: '10^{-9}',
        1e-10: '10^{-10}'
    }

    for key, value in result.items():
        if value in replacements:
            result[key] = replacements[value]
        elif isinstance(value, int) and value >= 1000:
            result[key] = f'{value:,}'

    with open(args.result, 'w', encoding='utf-8') as file_:
        for key, value in result.items():
            print(f'\def\{key}{{{value}}}', file=file_)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
