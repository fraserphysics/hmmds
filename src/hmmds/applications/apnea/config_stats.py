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
import os

import numpy
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
    """Holds properties of records and method to calculate them

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

        path = os.path.join(args.root, 'raw_data/apnea/summary_of_training')
        self.minute2class = utilities.read_expert(path, record_name)

    def normalize(self: Record, norm_avg):
        """Make PSD sum of this record match others"""
        self.raw_hr *= norm_avg / self.pass1.statistic_2()

    def smooth(self: Record):
        """Assign self.smoothed
        """
        # pylint: disable=attribute-defined-outside-init, invalid-name
        self.fft_length = 131072
        low_pass_width = 2 * numpy.pi / self.args.low_pass_period
        self.RAW_HR = numpy.fft.rfft(self.raw_hr, self.fft_length)
        self.smoothed = numpy.fft.irfft(
            utilities.window(self.RAW_HR, 1 / self.hr_sample_frequency,
                             0 * PINT('Hz'), low_pass_width))[:len(self.raw_hr)]

    def find_peaks_intervals(
            self: Record,
            min_prominence,
            peak_dict,
            distance=0.417 * PINT('minutes'),
            wlen=1.42 * PINT('minutes'),
    ):
        """ Put (peak, interval) pairs in peak_dict[class] for each detected peak.

        Args:
            min_prominence: Minimum for detected peaks
            distance: Minimum time between peaks
            wlen: Window length

        """
        s_f_hz = self.hr_sample_frequency.to('Hz').magnitude
        s_f_cpm = int(s_f_hz * 60)
        distance_samples = distance.to('seconds').magnitude * s_f_hz
        wlen_samples = wlen.to('seconds').magnitude * s_f_hz

        peak_times, properties = scipy.signal.find_peaks(
            self.smoothed,
            distance=distance_samples,
            prominence=min_prominence,
            wlen=wlen_samples)

        for index in range(len(peak_times) - 1):
            t_peak = peak_times[index]
            prominence_t = properties['prominences'][index]
            period_t = (peak_times[index + 1] - t_peak) / s_f_cpm
            t_minute = t_peak // s_f_cpm
            if t_minute >= len(self.minute2class):
                break
            class_t = self.minute2class[t_minute]
            peak_dict[class_t].append((prominence_t, period_t))


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    # setattr uses local variables pylint: disable=possibly-unused-variable
    min_prominence = args.min_prominence

    if args.records:
        record_names = args.records
    else:
        record_names = args.a_names

    records = dict((record_name, Record(args, record_name))
                   for record_name in record_names)

    if args.normalize:
        norm_sum = 0.0
        for record in records.values():
            norm_sum += record.pass1.statistic_2()
        norm_avg = norm_sum / len(record_names)
        for record in records.values():
            record.normalize(norm_avg)
        normalize = True
    else:
        normalize = False
        norm_avg = None

    # peak_dict holds (prominence, interval) pairs
    peak_dict = {0: [], 1: []}
    for record in records.values():
        record.smooth()
        record.find_peaks_intervals(record.args.min_prominence, peak_dict)

    limit = 2.2  # No intervals longer than this for pdf ratio fit
    sigma = 0.1  # Kernel width
    _lambda = 0.06  # Regularize pdf ratio fit
    # create a density_ratio.DensityRatio instance
    pdf_ratio = utilities.make_density_ratio(peak_dict, limit, sigma, _lambda)

    normal_pdf_spline = scipy.interpolate.CubicSpline(pdf_ratio.x,
                                                      pdf_ratio.y,
                                                      bc_type='natural',
                                                      extrapolate=True)

    result = argparse.Namespace()
    result.apnea_pdf = utilities.apnea_pdf
    result.normal_pdf = utilities.normal_pdf
    local = locals()
    for key in '''normalize min_prominence norm_avg
    normal_pdf_spline pdf_ratio'''.split():
        setattr(result, key, local[key])
    with open(args.result_path, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
