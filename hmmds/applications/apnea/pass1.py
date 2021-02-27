"""Make text file with lines like


b04 # Medium stat=  2.178 llr= -0.140 R=  2.248
a09 # High   stat=  4.263 llr=  4.162 R=  2.182
x33 # Medium stat=  1.847 llr= -0.322 R=  2.008
"""
import sys
import os
import argparse
import glob
import pickle

import numpy

import hmmds.applications.apnea.utilities


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


def log_likelihood_ratio(data, normal_model, apnea_model) -> float:
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
    return (a_ll - n_ll) / n_times


def make_reports(args, names: list):
    """Make a list of reports for pass1

    Args:

        args: heart_rate (path
            to heart rate data), respiration (path to respiration
            data), low_line (threshold for first pass classification),
            high_line (threshold for first pass classification)

        names: Eg, [

    """
    with open(os.path.join(args.models, args.Amodel), 'rb') as _file:
        apnea_model = pickle.load(_file)
    with open(os.path.join(args.models, args.BCmodel), 'rb') as _file:
        normal_model = pickle.load(_file)
    reports = []
    for name in names:
        y_data = hmmds.applications.apnea.utilities.heart_rate_respiration_data(
            os.path.join(args.heart_rate, name),
            os.path.join(args.respiration, name))
        # y_data is a dict
        r = r_stat(y_data['filtered_heart_rate_data'])
        llr = log_likelihood_ratio(y_data, normal_model, apnea_model)
        stat = r + args.stat_slope * llr
        if stat < args.low_line:
            level = 'Low'
        elif stat > args.high_line:
            level = 'High'
        else:
            level = 'Medium'
        reports.append(hmmds.applications.apnea.utilities.Pass1Item(name, llr, r, stat, level))
    return reports


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser("Create and write/pickle pass1_report")
    hmmds.applications.apnea.utilities.common_args(parser)
    args = parser.parse_args(argv)

    def get_names(letter):
        return [
            os.path.basename(x)
            for x in glob.glob('{0}/{1}*'.format(args.heart_rate, letter))
        ]

    reports = make_reports(
        args,
        get_names('a') + get_names('b') + get_names('c') + get_names('x'))
    reports.sort(key=lambda x: x.stat)
    with open(args.pass1 + '.pickle', 'wb') as _file:
        pickle.dump(reports, _file)
    with open(args.pass1, 'w') as _file:
        for report in reports:
            _file.write(
                '{0} # {1:6s} stat={2:6.3f} llr={3:6.3f} R={4:6.3f}\n'.format(
                    report.name, report.level, report.stat, report.llr,
                    report.r))


if __name__ == "__main__":
    sys.exit(main())
