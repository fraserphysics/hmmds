"""laser_values.py Read computation parameters write in form for LaTeX

"""
from __future__ import annotations  # Enables, eg, (self: LocalNonStationary

import sys
import argparse
import pickle
import os

import numpy


def parse_args(argv, arg_list):
    """Parse a command line.

    Args:
        argv:
        arg_list: Read values from files named in arg_list
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
def LaserLP5(dict_in):
    result = {}
    for key in 't_start t_stop'.split():
        result[f"laserLaserLP5{key.replace('_', '')}"] = dict_in[key]
    return result


@register
def LaserLogLike(dict_in):
    result = {}
    for key in 't_start t_stop'.split():
        result[f"laserLaserLogLike{key.replace('_', '')}"] = dict_in[key]
    return result


@register
def LaserStates(dict_in):
    result = {}
    for key in 't_start t_stop'.split():
        result[f"laserLaserStates{key.replace('_', '')}"] = dict_in[key]
    return result


@register
def LaserForecast(dict_in):
    result = {}
    for key in 't_start t_stop'.split():
        result[f"laserLaserForecast{key.replace('_', '')}"] = dict_in[key]
    return result


@register
def LaserHist(dict_in):
    result = {}
    for key in 'n_bins t_start t_stop'.split():
        result[f"laserLaserHist{key.replace('_', '')}"] = dict_in[key]
    return result


def main(argv=None):
    """Study dependence of relative entropy of Kalman filters on
    actual sample time and assumed observation noise.

    """
    if argv is None:
        argv = sys.argv[1:]
    # List of targets in Rules.mk
    arg_list = 'LaserHist LaserLP5 LaserLogLike LaserStates LaserForecast'.split(
    )
    args = parse_args(argv, arg_list)

    result = {}

    for name in arg_list:
        path = getattr(args, name)
        with open(path, 'rb') as file_:
            dict_in = pickle.load(file_)
        result.update(FunctionDict[name](dict_in))

    with open(args.result, 'w', encoding='utf-8') as file_:
        for key, value in result.items():
            print(f'\def\{key}{{{value}}}', file=file_)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
