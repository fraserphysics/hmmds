""" rtimes2hr.py:  Calculate low-pass filtered heart rate

python rtimes2hr.py rtimes_dir low_pass_heart_rate_dir a01 b01 c01 x01 etc

For each record R listed, read rtimes_dir/R.rtimes, calulate the
lowpass filtered heart rate and write low_pass_heart_rate_dir/R.lphr.
"""

import sys
import os.path
import argparse
import pickle

import pint
import numpy
import numpy.fft

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Calculate heart rate from Rtimes')
    parser.add_argument('--samples_per_minute', type=int, default=10)
    parser.add_argument('Rtimes_dir', type=str, help='Path to Rtimes data')
    parser.add_argument('lphr_dir',
                        type=str,
                        help='Path to low pass heart rate data')
    parser.add_argument('record_names',
                        type=str,
                        help='EG, a01 a02 b01 b02 x01',
                        nargs='+')
    return parser.parse_args(argv)


def calculate(rtimes: numpy.ndarray, args) -> dict:
    """ Calculate heart rate and low pass heart rate

    Args:
        rtimes: Times in seconds
    """

    # Find median and extreams of the intervals between rtimes
    intervals = rtimes[1:] - rtimes[:-1]
    _sorted = numpy.sort(intervals)
    n_sorted = len(_sorted)
    top = 1.25 * _sorted[int(0.95 * n_sorted)]
    bottom = 0.8 * _sorted[int(0.05 * n_sorted)]
    median = _sorted[n_sorted // 2]

    # Make clean, a version of intervals with extreme values replaced
    # by median
    clean = numpy.empty((len(rtimes),)) * PINT('s')
    clean[:-1] = numpy.where(numpy.logical_and(bottom < intervals, intervals < top), intervals, median)
    clean[-1] = clean[-2]

    # Create periods, an array of periods that is uniformly sampled at
    # frequency

    frequency = 2 * PINT('Hz')
    n_times = int(frequency * rtimes[-1])
    # Number of samples of inverse heart rate
    periods = numpy.empty(n_times) * PINT('s')
    t_old = rtimes[0]
    t_new = rtimes[1]
    period_old = clean[0]
    period_new = clean[1]
    i_rtime = 1  # index of rtimes and clean
    # t_old        t_new
    #   |            |
    #------------------------------------------
    #        |
    #     t_period
    for k_period in range(n_times):
        # Assign a value to periods[k_period] for every k_period
        t_period = k_period / frequency
        # Move t_old and t_new forward till t_period is in range
        while t_period > t_new:
            i_rtime += 1
            if i_rtime >= n_sorted:
                break
            t_old = t_new
            period_old = period_new
            t_new = rtimes[i_rtime]
            period_new = clean[i_rtime]
        # Calculate slope of period between t_old and t_new
        drdt = (period_new - period_old) / (t_new - t_old)
        period = period_old + (t_period - t_old) * drdt
        periods[k_period] = period

    # Transform periods to heart rate
    hr = 1.0 / periods
    hr_mean = hr.mean()
    hr_mean_0 = hr - hr_mean
    # Now, hr is heart rate sampled at 2HZ, and hrL is the same with
    # mean subtracted

    # 131072 is 2^17 which makes fft fast.  At 2Hz it is 18.2 hours
    # which is more than twice as long as any of the sleep records.
    # So rfft will pad with zeros.
    HR = numpy.fft.rfft(hr_mean_0.to('Hz').magnitude, 131072)
    HR[4000:] *= 0  # Drop frequencies above (4000*60)/65536 about 3.66/min
    hr_low_pass = numpy.fft.irfft(HR)[:n_times] * PINT('Hz') + hr_mean
    HR[0:100] *= 0  # Drop frequencies below (100*60)/65536 about 0.09/min
    hr_band_pass = numpy.fft.irfft(HR)[:n_times] * PINT('Hz')
    # Write results in beats per minute
    result = {
        'sample_frequency': frequency,
        'hr': hr.to('1/minute'),
        'hr_low_pass': hr_low_pass.to('1/minute'),
        'hr_band_pass': hr_band_pass.to('1/minute'),
    }
    return result


def read_rtimes(path):
    with open(path, mode='r', encoding='utf-8') as _file:
        lines = _file.readlines()
    n_times = len(lines)
    rtimes = numpy.empty(n_times)
    for t, line in enumerate(lines):
        rtimes[t] = float(line)
    return rtimes * PINT('second')


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    for name in args.record_names:
        rtimes = read_rtimes(os.path.join(args.Rtimes_dir, name + '.rtimes'))
        hr_dict = calculate(rtimes, args)
        with open(os.path.join(args.lphr_dir, name + '.lphr'), 'wb') as _file:
            pickle.dump(hr_dict, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
