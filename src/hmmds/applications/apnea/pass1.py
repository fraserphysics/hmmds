"""For every record calculate two statistics for classifying the record

All paths are derived from root via utilities.Common

The result is written both as a pickle file and a text file with lines
like


b04 # Medium stat=  2.178 llr= -0.140 R=  2.248
a09 # High   stat=  4.263 llr=  4.162 R=  2.182
x33 # Medium stat=  1.847 llr= -0.322 R=  2.008

"""
import sys
import os
import argparse
import glob
import pickle
import typing

import numpy

import hmmds.applications.apnea.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser("Create and write/pickle pass1_report")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def r_stat(data: numpy.ndarray) -> float:
    """ Calculate the statistic R to help classify records

    Args:
        data: The scalar time series of filtered heart rate data

    Returns:
        The height of peak at the 74% level / the average peak height
    """
    n_times = len(data)
    peaks = []
    window_size = 5  # Window is =/- window_size
    for t in range(window_size, n_times - window_size):
        argmax_window = data[t - window_size:t + window_size].argmax()
        # Look for positive peak centered in the window
        if argmax_window == window_size and data[t] > 0:
            peaks.append(data[t])
    peaks.sort()
    # return 74% level / average L1 norm of data per sample
    return peaks[int(.74 * len(peaks))] / (numpy.abs(data).sum() / n_times)


def log_likelihood_ratio(data, normal_model,
                         apnea_model) -> typing.Tuple[float, float, int]:
    """Calculate the ratio of the likelihoods for two models

    Args:
        data: Combination of heart rate and respiration data
        normal_model: HMM trained on normal data
        apnea_model: HMM trained on apnea data

    Returns:
        (log_likelihood(apnea_model) - log_likelihood(normal_model) /
            (length of data)

    """

    def log_likelihood(hmm):
        t_seg = hmm.y_mod.observe([data])
        assert len(t_seg) == 2
        hmm.calculate()
        return hmm.forward(), t_seg[-1]

    a_ll, a_times = log_likelihood(apnea_model)
    n_ll, n_times = log_likelihood(normal_model)
    assert a_times == n_times
    return a_ll, n_ll, n_times


def make_reports(args, names: list):
    """Make a list of reports for pass1

    Args:

        args: paths and parameters from command line and utilities.py

        names: Eg, 'a01 a02 x34'.split()

    """
    with open(args.Amodel, 'rb') as _file:
        apnea_model = pickle.load(_file)
    with open(args.BCmodel, 'rb') as _file:
        normal_model = pickle.load(_file)
    reports = {}
    for name in names:
        y_data = hmmds.applications.apnea.utilities.heart_rate_respiration_data(
            name, args)
        # y_data is a dict
        a_ll, n_ll, n_times = log_likelihood_ratio(y_data, normal_model,
                                                   apnea_model)
        reports[name] = {
            'a_ll': a_ll,
            'n_ll': n_ll,
            'n_times': n_times,
        }
    return reports


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    reports = make_reports(args, args.all_names)
    with open(args.pass1 + '.pickle', 'wb') as _file:
        pickle.dump(reports, _file)
    with open(args.pass1, 'w') as _file:
        for name, report in reports.items():
            a_per = report['a_ll'] / report['n_times']
            n_per = report['n_ll'] / report['n_times']
            _file.write(
                f'{name} # a: {a_per:6.3f} n: {n_per:6.3f} a-n: {a_per-n_per:6.3f}'
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
