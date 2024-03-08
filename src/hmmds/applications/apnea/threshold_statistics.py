"""threshold_statistics.py Explore dependence of threshold on statistics

python threshold_statistics.py default threshold_statistics.pdf a01 a02 ...

Derived from shift_threshold.py
"""
from __future__ import annotations

import sys
import argparse
import typing
import pickle

import numpy
import scipy.optimize

import utilities
import plotscripts.utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Plot to check if statistics predict best threshold")
    utilities.common_arguments(parser)
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('c_model_path', type=str, help="path to model")
    parser.add_argument('a_model_path', type=str, help="path to model")
    parser.add_argument('result_path', type=str, help="path to pdf file")
    parser.add_argument('record_names', type=str, nargs='+')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


class ModelRecord(utilities.ModelRecord):

    def __init__(self: ModelRecord, model_path: str, record_name: str):
        utilities.ModelRecord.__init__(self, model_path, record_name)
        self.pass1 = utilities.Pass1(record_name, self.model.args).statistic_1()
        self.threshold, self.counts = self.best_threshold()

    def log_likelihood(self: ModelRecord):
        """ Calculate log P(data|model)
        """
        class_model = self.model.y_mod['class']
        del self.model.y_mod['class']
        p_yt_steps = self.model.likelihood(self.y_raw_data[0])
        self.model.y_mod['class'] = class_model  # Restore for future use
        result = numpy.log(p_yt_steps).sum()
        return result


def fit(names: list, a_model_record: dict, c_model_record: dict):
    """Fit affine model for threshold
    Args:
        a_model_record: Eg, a_model_record["a01"].model was fit to APLUSNAMES
        c_model_record: Eg, c_model_record["a01"].model is not fit to data
    
    Return: coefficients, fit_thresholds

    Use numpy.linalg.lstsq(a,b) to compute the vector x that
    approximately solves the equation a @ x = b where

    a[i,j] is for record i and j = [Pass1, log_like_a, log_like_c, 1]

    b[i] is log(best_threshold) for record i

    The returned coefficients is the vector x.  The returned
    thresholds are the values of a @ x

    """
    a = numpy.empty((len(names), 4))
    b = numpy.empty(len(names))
    for i, name in enumerate(names):
        a[i, 0] = a_model_record[name].pass1
        a[i, 1] = a_model_record[name].log_likelihood()
        a[i, 2] = c_model_record[name].log_likelihood()
        a[i, 3] = 1
        b[i] = numpy.log(a_model_record[name].threshold)
    x = numpy.linalg.lstsq(a, b, rcond=None)[0]
    thresholds = numpy.exp(a @ x)
    return x, thresholds


def main(argv=None):
    """Plot best thresholds on training data against various statistics

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    fig, (pass1_axes, like_a_axes, like_c_axes,
          fit_axes) = pyplot.subplots(nrows=4, ncols=1, figsize=(8, 4))

    counts_sum = 0
    a_model_record = {}
    c_model_record = {}
    for name in args.record_names:
        a_model_record[name] = ModelRecord(args.a_model_path, name)
        counts_sum += a_model_record[name].counts

        c_model_record[name] = ModelRecord(args.c_model_path, name)

    fraction = (counts_sum[1] + counts_sum[2]) / counts_sum.sum()
    coefficients, fit_thresholds = fit(args.record_names, a_model_record,
                                       c_model_record)
    pairs = list(
        zip('pass1 log_like_a log_like_c 1'.split(),
            (f'{x:4.2g}' for x in coefficients)))
    print(f'''
{counts_sum=} errors={counts_sum[1]+counts_sum[2]} {fraction=}
{pairs}
''')
    for name, fit_threshold in zip(args.record_names, fit_thresholds):
        x = a_model_record[name].pass1
        y = a_model_record[name].threshold
        pass1_axes.semilogy(x,
                            y,
                            marker=f'${name}$',
                            markersize=14,
                            linestyle='None')
        pass1_axes.set_xlabel('pass1')

        x = a_model_record[name].log_likelihood()
        like_a_axes.semilogy(x,
                             y,
                             marker=f'${name}$',
                             markersize=14,
                             linestyle='None')
        like_a_axes.set_xlabel('like_a')

        x = c_model_record[name].log_likelihood()
        like_c_axes.semilogy(x,
                             y,
                             marker=f'${name}$',
                             markersize=14,
                             linestyle='None')
        like_c_axes.set_xlabel('-like_c')

        fit_axes.loglog(fit_threshold,
                        y,
                        marker=f'${name}$',
                        markersize=14,
                        linestyle='None')
        fit_axes.set_xlabel('fit')
    if args.show:
        pyplot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
