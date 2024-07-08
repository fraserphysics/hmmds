"""fft_threshold_statistics.py Calculate statistics of records and map to threshold

python fft_threshold_statistics.py $(MODELS)/default $@ $(APLUSNAMES)

Write pickle file containing:

    args.record_names: Records used for fit

    best_thresholds: Dict of thresholds for each a, b, c record.

    fit_thresholds: Dict of fit thresholds for each a, b, c, x record

    Leave one out error as a function of SVD rank

"""
from __future__ import annotations

import sys
import argparse
import typing
import pickle

import numpy
import numpy.linalg
import scipy.signal
import sklearn.svm
import sklearn.pipeline
import sklearn.preprocessing

import utilities
import hmmds.applications.apnea.model_init


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Fit function for threshold of each record")
    utilities.common_arguments(parser)
    parser.add_argument('--resolution',
                        type=float,
                        nargs=3,
                        default=(-3.0, 3.0, 100),
                        help="geometric range of thresholds")
    parser.add_argument('--result_path', type=str, help="path to pickle file")
    parser.add_argument('--leave_one_out',
                        action='store_true',
                        help='In turn, hold each record out from fit')
    parser.add_argument('--all_in',
                        action='store_true',
                        help='Report performance on each training record')
    parser.add_argument('classification_model_path',
                        type=str,
                        help="path to model")
    parser.add_argument('record_names', type=str, nargs='+')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    args.low = float(args.resolution[0])
    args.high = float(args.resolution[1])
    args.levels = int(args.resolution[2])
    return args


class ModelRecord(utilities.ModelRecord):

    def __init__(self: ModelRecord, args, record_name: str):
        """

        Args:
            model_path: Path to HMM.  Observation models include 'class'
            record_name: EG, 'a01'
            psd: For estimating threshold
        """
        model_path = args.classification_model_path
        utilities.ModelRecord.__init__(self, model_path, record_name)
        # Read the raw heart rate
        path = args.heart_rate_path_format.format(record_name)
        with open(path, 'rb') as _file:
            _dict = pickle.load(_file)
        assert set(_dict.keys()) == set('hr sample_frequency'.split())
        assert _dict['sample_frequency'].to('Hz').magnitude == 2

        self.hr_sample_frequency = _dict['sample_frequency']
        self.raw_hr = _dict['hr'].to('1/minute').magnitude
        self.threshold = None
        self.second_derivative = None

    def get_best_threshold(self: ModelRecord, low, high, levels):
        """Calculate and return best threshold.

        Also assign to self.second_derivative estimate of d^2 / d threshold^2

        Here threshold is log_10(x) where x is used at the lowest level
        """
        if self.threshold is None:
            self.threshold, self.counts = self.best_threshold(low, high, levels)
            exp_threshold = 10.0**self.threshold
            self.classify(threshold=exp_threshold * 10)
            plus_counts = self.score()
            self.classify(threshold=exp_threshold / 10)
            minus_counts = self.score()
            self.second_derivative = (plus_counts[1:3].sum(
            ) + minus_counts[1:3].sum()) / 1 - self.counts[1:3].sum()

        return self.threshold


def psd(args, name):
    """Estimate the power spectral density for named record
    """
    with open(args.heart_rate_path_format.format(name), 'rb') as _file:
        _dict = pickle.load(_file)
        sample_frequency = _dict['sample_frequency'].to('1/minutes').magnitude
        hr = _dict['hr']
        trim = int(sample_frequency * args.trim_start.to('minutes').magnitude)
    heart_rate = _dict['hr'].to('1/minute').magnitude[trim:-trim]
    frequencies, psd_ = scipy.signal.welch(heart_rate,
                                           fs=sample_frequency,
                                           nperseg=args.fft_width)
    return psd_, frequencies


def svm(psd_dict, a_names, c_names):
    """Find maximum margin separating hyperplane for pass1
    """

    n_samples = len(a_names) + len(c_names)
    n_features = len(psd_dict[a_names[0]])
    X = numpy.empty((n_samples, n_features))
    y = numpy.empty(n_samples)
    class_dict = {'a': 1, 'c': 0}
    for i, name in enumerate(a_names + c_names):
        X[i, :] = psd_dict[name]
        y[i] = class_dict[name[0]]
    clf = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        sklearn.svm.LinearSVC(random_state=0, tol=1e-5, dual=False))
    clf.fit(X, y)

    coefficients = clf.named_steps['linearsvc'].coef_
    return coefficients


def leave_one_out(args, model_records, psd_dict):
    """For each record in model_records, fit leaving that record out
    and test on that record.

    """
    for rank in range(1, len(args.record_names)):
        error_sum = 0
        for name in args.record_names:
            names = args.record_names.copy()
            names.remove(name)
            fit = Fit(names, model_records, rank, psd_dict)
            threshold = fit.threshold(psd_dict[name])
            model_records[name].classify(threshold=10**threshold)
            counts = model_records[name].score()
            error_sum += counts[1:3].sum()
        print(f'{rank=:2d} {error_sum=}')


class Fit:

    def __init__(self: Fit, names, model_records, rank, psd_dict):
        """Fit linear model for threshold

        Args:
            args: 
        
        Use numpy.linalg.lstsq(a,b) to compute the vector x that
        approximately solves the equation a @ x = b where

        a[i,:] = psd for record i

        b[i] is log(best_threshold) for record i


        """

        self.fit = {}

        n_names = len(names)
        n_frequencies = len(psd_dict[names[0]])
        psds = numpy.empty((n_names, n_frequencies))
        t_best = numpy.empty(n_names)
        H = numpy.zeros((n_names, n_names))
        for i, name in enumerate(names):
            psds[i, :] = numpy.log10(psd_dict[name])
            t_best[i] = model_records[name].threshold
            H[i, i] = model_records[name].second_derivative

        U, S, VT = numpy.linalg.svd(H @ psds)
        error = f'{U.shape=} {S.shape=} {VT.shape=}'
        assert U.shape == (n_names, n_names), error
        assert S.shape == (n_names,), error
        assert VT.shape == (n_frequencies, n_frequencies), error
        '''The rows of VT are directions in PSD space.  The columns of
        U are directions in record space.

        '''
        self.coefficients = VT.T[:, :rank] @ (U[:, :rank] /
                                              S[:rank]).T @ H @ t_best
        assert self.coefficients.shape == (n_frequencies,)

        for name in names:
            self.fit[name] = self.threshold(psd_dict[name])

    def threshold(self: Fit, psd_):
        """Calculate estimated threshold as linear function of
        a psd.

        """
        high = 3.0
        low = -3.0
        result = numpy.log10(psd_) @ self.coefficients
        return min(high, max(low, result))


def all_in(args, model_records, fit):
    """ Print error counts for records in args.record_names
    """
    total_errors = 0
    print('''In fit_threshold_statistics.all_in
name best  fit   difference N2N N2A A2N A2A''')
    for name in args.record_names:
        best = model_records[name].threshold
        _fit = fit.fit[name]
        model_records[name].classify(threshold=10**_fit)
        counts = model_records[name].score()
        total_errors += counts[1:3].sum()
        print(
            f'{name} {best:5.2f} {_fit:5.2f} {best-_fit:5.2f}       {counts[0]:3d} {counts[1]:3d} {counts[2]:3d} {counts[3]:3d}'
        )
    print(f'{total_errors=}')


def calculate_statistics(args, model_records, fit, psd_dict, frequencies):
    """Create and return a dict of statistics containing:

    result['args'] Command line args
    result['best_threshold'] Best threshold for each of the 70
        records.  Obtained by cheating for training records.  Value is
        0.0 for x-files
    result['psds'] A psd for each of the 70 records
    result['frequencies'] The frequencies for each bin in the psds
    result['threshold_coeficients'] v with threhsold[name] = v @ psd[name]
    result['pass1_coefficients'] v @ psd > pass1_threshold -> A or N

    """

    result = {}
    result['args'] = args
    best_thresholds = result['best_threshold'] = {}
    result['psds'] = psd_dict
    result['frequencies'] = frequencies
    for name in args.all_names:
        if name[0] == 'x':
            best_thresholds[name] = 0.0
        else:
            best_thresholds[name] = model_records[name].get_best_threshold(
                args.low, args.high, args.levels)
        assert isinstance(best_thresholds[name], float)
        #print(f"{name} {result['threshold']['best'][name]=:5.2f} {result['threshold']['fit'][name]=:5.2f}")
    result['threshold_coefficients'] = fit.coefficients
    a_names = []
    c_names = []
    for name in model_records.keys():
        if name[0] == 'a':
            a_names.append(name)
        if name[0] == 'c':
            c_names.append(name)
    c_names.remove('c04')  # Severe arrhythmia in c04
    result['pass1_coefficients'] = svm(psd_dict, a_names, c_names)
    return result


def main(argv=None):
    """Calculate various statistics and parameters for f(record) ->
    threshold, and write to a pickle file

    """

    numpy.seterr(divide='raise', invalid='raise')
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    test_names = args.a_names + args.b_names + args.c_names

    # Calculate a psd for each record.  Some might be unnecessary, but
    # they are cheap to calculate.
    psds = dict((name, psd(args, name)[0]) for name in args.all_names[1:])
    first_name = args.all_names[0]
    psds[first_name], frequencies = psd(args, first_name)

    # Make a ModelRecord for each record with expert annotations.  It
    # is expensive because of the search for the best threshold.
    model_records = {}
    for name in test_names:
        model_records[name] = ModelRecord(args, name)
        # Assign threshold, counts, and second_derivative to model_records[name]
        model_records[name].get_best_threshold(args.low, args.high, args.levels)

    if args.leave_one_out:
        leave_one_out(args, model_records, psds)
    if args.all_in or args.result_path:
        fit = Fit(args.record_names, model_records, len(args.record_names),
                  psds)
    if args.all_in:
        all_in(args, model_records, fit)
    if args.result_path:
        statistics = calculate_statistics(args, model_records, fit, psds,
                                          frequencies)
        with open(args.result_path, 'wb') as _file:
            pickle.dump(statistics, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
