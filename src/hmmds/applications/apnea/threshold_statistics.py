"""threshold_statistics.py Calculate statistics of records for map to threshold

python threshold_statistics.py c_model default result.pkl a01 a02 ...

"""
from __future__ import annotations

import sys
import argparse
import typing
import pickle

import numpy

import utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Fit function for threshold of each record")
    utilities.common_arguments(parser)
    parser.add_argument('--resolution',
                        type=float,
                        nargs=3,
                        default=(1.0e-4, 1.0e4, 10),
                        help="geometric range of thresholds")
    parser.add_argument('a_model_path', type=str, help="path to model")
    parser.add_argument('high_model_path', type=str, help="path to model")
    parser.add_argument('low_model_path', type=str, help="path to model")
    parser.add_argument('result_path', type=str, help="path to pickle file")
    parser.add_argument('record_names', type=str, nargs='+')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    args.low = float(args.resolution[0])
    args.high = float(args.resolution[1])
    args.levels = int(args.resolution[2])
    return args


class ModelRecord(utilities.ModelRecord):

    def __init__(self: ModelRecord, model_path: str, record_name: str):
        utilities.ModelRecord.__init__(self, model_path, record_name)

        # Calculate statistics
        self.pass1 = utilities.Pass1(record_name, self.model.args).statistic_1()

        if self.has_class:
            class_model = self.model.y_mod['class']  # Save
            del self.model.y_mod['class']
            p_yt_steps = self.model.likelihood(self.y_raw_data[0])
            self.model.y_mod['class'] = class_model  # Restore for future use
        else:
            p_yt_steps = self.model.likelihood(self.y_raw_data[0])
        self.log_likelihood = numpy.log(p_yt_steps).sum()

    def get_threshold(self, low, high, levels):
        self.threshold, self.counts = self.best_threshold(low, high, levels)
        return self.threshold


class Fit:

    def __init__(self, names: list, a_model_record: dict,
                 high_model_record: dict, low_model_record: dict, low, high,
                 levels):
        """Fit affine model for threshold
        Args:
            names: Records to use for fitting parameters
            a_model_record: Eg, a_model_record["a01"].model was fit to APLUSNAMES
            high_model_record: Eg, high_model_record["a01"].model was fit records with high thresholds
            low_model_record: Eg, high_model_record["a01"].model was fit records with low thresholds
        
        Use numpy.linalg.lstsq(a,b) to compute the vector x that
        approximately solves the equation a @ x = b where

        a[i,j] is for record i and j = [Pass1, log_like_a, log_like_c, 1]

        b[i] is log(best_threshold) for record i


        """
        a = numpy.empty((len(names), 4))
        b = numpy.empty(len(names))
        for i, name in enumerate(names):
            a[i, 0] = a_model_record[name].pass1
            a[i, 1] = high_model_record[name].log_likelihood
            a[i, 2] = low_model_record[name].log_likelihood
            a[i, 3] = 1
            b[i] = numpy.log(a_model_record[name].get_threshold(
                low, high, levels))
        self.coefficients = numpy.linalg.lstsq(a, b, rcond=None)[0]

    def threshold(self, a_model_record, high_model_record, low_model_record):
        """Calculate estimated threshold as affine function of
        statistics of a record

        """
        a = numpy.empty(4)
        a[0] = a_model_record.pass1
        a[1] = high_model_record.log_likelihood
        a[2] = low_model_record.log_likelihood
        a[3] = 1
        return numpy.exp(a @ self.coefficients)


def main(argv=None):
    """Calculate various statistics and parameters for f(record) ->
    threshold, and write to a pickle file

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    a_model_record = {}
    high_model_record = {}
    low_model_record = {}
    for name in args.all_names:
        a_model_record[name] = ModelRecord(args.a_model_path, name)
        high_model_record[name] = ModelRecord(args.high_model_path, name)
        low_model_record[name] = ModelRecord(args.low_model_path, name)

    fit = Fit(args.record_names, a_model_record, high_model_record,
              low_model_record, args.low, args.high, args.levels)

    statistics = {}
    for name in args.all_names:
        pass1 = a_model_record[name].pass1
        high_log_likelihood = high_model_record[name].log_likelihood
        low_log_likelihood = low_model_record[name].log_likelihood
        fit_threshold = fit.threshold(
            a_model_record[name],
            high_model_record[name],
            low_model_record[name],
        )
        if name[0] == 'x':
            best_threshold = 1.0
        else:
            best_threshold = a_model_record[name].get_threshold(1e-4, 1e4, 10)
        statistics[name] = numpy.array([
            pass1, high_log_likelihood, low_log_likelihood, 1.0, fit_threshold,
            best_threshold
        ])
    with open(args.result_path, 'wb') as _file:
        pickle.dump((args.record_names, fit, statistics), _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
