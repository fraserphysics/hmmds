"""config_stats.py: Write a pickle file with statistics of
peaks in the heart rate signal

python config_stats.py 6.0 config.pkl

Characteristics are:

apnea_pdf           Probability density function for inter-peak intervals

normal_pdf          Probability density function for inter-peak intervals

min_prominence      Minimum prominence for detcting a peak

boundaries          For binning prominences of peaks

norm_avg            Average over records of Pass1.statistic_2()

n_per_bin           Number of peaks in the training data in each bin

normal_pdf_spline   Called by utilities.normal_pdf

pdf_ratio           Attributes x and y used in utilities.normal_pdf

"""
import sys
import pickle
import argparse

import numpy
import scipy.interpolate

import utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Analyze peaks of heart rate sign")
    utilities.common_arguments(parser)
    parser.add_argument(
        '--normalize',
        type=str,
        help=
        "Path to config that has norm_avg for dividing heart rate signal by PSD sum"
    )
    parser.add_argument(
        '--n_bins',
        type=int,
        default=6,
        help='Number of quantization levels for peaks',
    )
    parser.add_argument(
        'min_prominence',
        type=float,
        help='Threshold for detecting heart rate peaks in beats per minute')
    parser.add_argument('result_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def analyze_record(args, record_name, peak_dict):
    raw_dict = utilities.read_slow_class(args, record_name)
    peaks, properties = utilities.peaks(raw_dict['slow'],
                                        args.heart_rate_sample_frequency,
                                        args.min_prominence)
    for peak, prominence in zip(peaks, properties['prominences']):
        peak_dict[raw_dict['class'][peak]].append(prominence)


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    min_prominence = args.min_prominence

    if args.normalize:
        # Set args.norm_avg to trigger normalization of the amplitude
        # of heart rate signals.  I believe that the normalization is
        # idempotent.
        with open(args.normalize, 'rb') as _file:
            old_config = pickle.load(_file)
        args.norm_avg = old_config.norm_avg

    if args.records:
        record_names = args.records
    else:
        record_names = args.a_names

    # Count peaks in minutes classified as apnea
    for record_name in record_names:
        heart_rate = utilities.read_slow_class(args, record_name)['slow']
        peaks, _ = utilities.peaks(heart_rate, args.heart_rate_sample_frequency,
                                   min_prominence)

    # peak_dict holds (prominence, interval) pairs
    peak_dict, boundaries, norm_avg = utilities.peaks_intervals(
        args, record_names)

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
    for key in '''min_prominence boundaries norm_avg
    normal_pdf_spline pdf_ratio'''.split():
        setattr(result, key, local[key])
    with open(args.result_path, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
