"""apnea_ratio.py Estimates the ratio of probability densities of
intervals between peaks for normal and apnea.  r(x) =
p_normal(x)/p_apnea(x).

python apnea_ratio.py --show pdf_ratio.pdf

I selected these values by experimenting and looking at plots:

limit = 2.2 The data is sparse for lengths > 2.  There is a peak in
            the estimated ratio of 5.0 at length = 1.7.  I belive
            these numbers are +/- 5%.  The cut-off value 2.2 is far
            enough away from the peak to not have much effect.  I
            avoid a large cut-off because it would increase the
            optimal sigma.

sigma = 0.1 Smaller values yield wiggles at the sampling frequency
            24/minute -> intervals are quantized with a step of
            0.041666...

lambda = 0.06 The densratio code says this is optimal

kernel_num = 800 Perhaps larger than necessary, but still fast.

I used apnea.plot_pp.py to create the file pickled_peak_dict.

See Density Ratio Estimation: A Comprehensive Review at
http://www.ms.k.u-tokyo.ac.jp/sugi/2010/RIMS2010.pdf

"""
import sys
import pickle
import argparse

import numpy

import utilities
import plotscripts.utilities
import density_ratio


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Estimate and plot p_normal()/p_apnea() for intervals between peaks of heart rate."
    )
    parser.add_argument('--pickle',
                        type=str,
                        help='',
                        default='pickled_peak_dict')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    utilities.common_arguments(parser)
    parser.add_argument('figure_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def plot(axes, pdf_ratio: density_ratio.DensityRatio, normal_pdf, apnea_pdf,
         max_interval: float):
    """The minimum value for interval is .4583333

    """
    z = numpy.linspace(0.1, max_interval, 1000).reshape(-1, 1)
    axes.plot(z, pdf_ratio(z), label='ratio')
    axes.plot(z, normal_pdf(z), label='normal')
    axes.plot(z, apnea_pdf(z), label='apnea')
    axes.set_xlabel('length/minute')
    axes.set_ylabel('p_normal/p_apnea')
    axes.legend()


def main(argv=None):
    """Estimate and plot ratio of probability densities for intervals
    between peaks.

    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    limit = 3.0
    sigma = 0.1
    _lambda = 0.06

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    fig, axes = pyplot.subplots(nrows=1, figsize=(6, 8))

    # Find peaks
    peak_dict, boundaries = utilities.peaks_intervals(args, args.a_names)
    with open(args.pickle, 'wb') as _file:
        pickle.dump(peak_dict, _file)

    interval_pdfs = utilities.make_interval_pdfs(args)
    plot(axes, interval_pdfs, interval_pdfs.normal_pdf, interval_pdfs.apnea_pdf,
         limit + 5 * sigma)

    if args.show:
        pyplot.show()
    fig.savefig(args.figure_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
