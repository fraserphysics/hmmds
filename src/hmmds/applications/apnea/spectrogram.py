""" spectrogram.py:  Calculate spectrogram from low pass heart rate files

python spectrogram.py a11 a11.sgram

Derived from respire.py
"""

from __future__ import annotations  # Enables, eg, (self: HMM,

import sys
import os.path
import argparse
import pickle
import typing

import pint
import numpy
import scipy.signal

from hmmds.applications.apnea import utilities

PINT = pint.get_application_registry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Calculate spectrogram from heart rate')
    utilities.common_arguments(parser)
    parser.add_argument('--sample_rate_out',
                        type=int,
                        default=10,
                        help='Samples per minute for results')
    parser.add_argument('--fft_width',
                        type=int,
                        default=128,
                        help='Number of samples for each fft')
    parser.add_argument('record_name', type=str, help="eg, 'a11'")
    parser.add_argument('result', type=str, help='Path for writing')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    args.sample_rate_out *= PINT('1/minutes')
    return args


def spectrogram(heart_rate: utilities.HeartRate, args):
    """Map a heart rate signal sampled at 2 Hz to a spectrogram

    Args:
        heart_rate: Holds estimated heart rate signal
        args: Command line arguments
    """

    heart_rate.filter_hr(
        resp_pass_center=17 / PINT('minute'),
        resp_pass_width=6 / PINT('minute'),
    )
    filtered_heart_rate = heart_rate.resp_pass
    time_series = heart_rate.get_slow() * PINT('1/minute')

    ratio = int(
        (args.model_sample_frequency / args.sample_rate_out).to('').magnitude)
    assert ratio == 12
    # With default args ratio is 12
    frequencies, times, psds = scipy.signal.spectrogram(
        filtered_heart_rate,
        fs=args.model_sample_frequency.to('Hz').magnitude,
        nperseg=args.fft_width,
        noverlap=args.fft_width - ratio,
        detrend=False,
        mode='psd')
    assert args.fft_width > len(filtered_heart_rate) - len(times) * ratio >= 0
    return {
        'frequencies': frequencies * PINT('Hz'),
        'times': times * PINT('second'),
        'psds': psds,
        'filtered_heart_rate': filtered_heart_rate,
        'time_series': time_series,
        'hr_dt': 1 / heart_rate.model_sample_frequency
    }


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    heart_rate = utilities.HeartRate(args, args.record_name)
    result = spectrogram(heart_rate, args)
    with open(args.result, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
