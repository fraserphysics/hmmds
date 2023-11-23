"""config_stats.py: Write a pickle file with statistics of
peaks in the heart rate signal

python config_stats.py 6.0 config.pkl

Characteristics are:

apnea_pdf           Probability density function for inter-peak intervals

normal_pdf          Probability density function for inter-peak intervals

normalize           Binary flag for normalizing data first

min_prominence      Minimum prominence for detcting a peak

norm_avg            The average over records of Pass1.statistic_2()

normal_pdf_spline   Called by utilities.normal_pdf

pdf_ratio           Attributes x and y used in utilities.normal_pdf

"""
from __future__ import annotations  # Enables, eg, self: Record

import sys
import pickle
import argparse

import scipy.interpolate
import pint

import utilities

PINT = pint.UnitRegistry()

# The following use config data and should not be used here:
# utilities.HeartRate, utilities.peaks_intervals


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Collect statistics about peaks of heart rate signal")
    utilities.common_arguments(parser)
    parser.add_argument(
        '--normalize',
        action='store_true',
        help="Make sum of all but low freq PSD channels match for each record",
    )
    parser.add_argument(
        'min_prominence',
        type=float,
        help='Threshold for detecting heart rate peaks in beats per minute')
    parser.add_argument('result_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


class Record:
    """Holds properties of records for calculating result.norm_avg

    """

    def __init__(self: Record, args, record_name):
        self.args = args
        # pass1.statistic_2, the sum of spectral power above 0.3 cpm,
        # is available for normalization
        self.pass1 = utilities.Pass1(record_name, args)

        # Read the raw heart rate
        path = args.heart_rate_path_format.format(record_name)
        with open(path, 'rb') as _file:
            _dict = pickle.load(_file)
        assert set(_dict.keys()) == set('hr sample_frequency'.split())
        assert _dict['sample_frequency'].to('Hz').magnitude == 2

        self.hr_sample_frequency = _dict['sample_frequency']
        self.raw_hr = _dict['hr'].to('1/minute').magnitude


def main(argv=None):
    """Write a pickle file with statistics of peaks in the heart rate
       signal

    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    if args.records:
        record_names = args.records
    else:
        record_names = args.a_names

    result = argparse.Namespace()

    if args.normalize:
        norm_sum = 0.0
        for record in (
                Record(args, record_name) for record_name in record_names):
            norm_sum += record.pass1.statistic_2()
        result.norm_avg = norm_sum / len(record_names)
        result.normalize = True
    else:
        result.normalize = False
        result.norm_avg = None

    result.min_prominence = args.min_prominence

    args.config = result  # Hack to enable use of utilities
    result.pdf_ratio = utilities.make_interval_pdfs(args, record_names)

    result.normal_pdf_spline = scipy.interpolate.CubicSpline(result.pdf_ratio.x,
                                                             result.pdf_ratio.y,
                                                             bc_type='natural',
                                                             extrapolate=True)

    result.apnea_pdf = utilities.apnea_pdf
    result.normal_pdf = utilities.normal_pdf
    with open(args.result_path, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
