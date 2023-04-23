"""states2hr.py: Map a decoded state sequence to a heart rate time
series

python states2hr.py states_file heart_rate_file

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
        description='Calculate heart rate from decoded states')
    parser.add_argument('--likelihood',
                        type=str,
                        help='path to time series of p(y[t]|y[:t]).')
    parser.add_argument('--censor',
                        type=float,
                        help='Fraction of data to censor')
    parser.add_argument('--samples_per_minute', type=int, default=10)
    parser.add_argument('--r_state',
                        type=int,
                        default=20,
                        help='state index for R peak')
    parser.add_argument('--outlier_state',
                        type=int,
                        default=0,
                        help='state index for outliers')
    parser.add_argument('states_file', type=str, help='Path to decoded states')
    parser.add_argument('heart_rate_file',
                        type=str,
                        help='Path to heart rate data')
    return parser.parse_args(argv)


def find_beats_and_periods(states: numpy.ndarray,
                           r_state=35,
                           outlier_state=0,
                           likelihood=None,
                           censor_fraction=0):
    """Args:
        states: State sequence from Viterbi decoding
        r_state: Defines beat time
        outlier_state: The state for ECG samples that are implausible.
        likelihood: p(y[t]|y[:t])
        censor_fraction: Fraction of intervals to reject based on likelihood

    Return: The pair (array of integer times and periods, set of bad
    intervals)

    Bad intervals contain atypical or implausible signal segments.
    Integer beat times at the start of such intervals tracks the set.

    """
    bad_intervals = set()

    r_indices = numpy.nonzero(states == r_state)[0]
    time_period = numpy.empty((len(r_indices) - 1, 2), dtype=int)
    time_period[:, 0] = r_indices[:-1]
    time_period[:, 1] = r_indices[1:] - r_indices[:-1]
    for (time, period) in time_period:
        if numpy.count_nonzero(states[time:time + period] == outlier_state) > 0:
            bad_intervals.add(time)
    if likelihood is None:
        assert censor_fraction == 0
        return time_period, bad_intervals

    period_likelihood = numpy.empty(len(time_period))
    for i, (time, period) in enumerate(time_period):
        period_likelihood[i] = likelihood[time:time + period].min()
    sorted_ = period_likelihood.copy()
    sorted_.sort()
    threshold = sorted_[int(censor_fraction * len(sorted_))]
    bad_intervals.update(
        time_period[numpy.nonzero(period_likelihood < threshold)[0], 0])
    return time_period, bad_intervals


def calculate(
    time_period: numpy.ndarray,
    bad_intervals: set,
    time_out,
    in_frequency=100 * PINT('Hz'),
    out_frequency=2 * PINT('Hz')) -> dict:
    """Calculate heart rate and low pass heart rate

    Args:
        time_period: Integer beat times and integer periods
        bad_intervals: Subset of beat times with untrusted subsequent intervals
        time_out: Pint time for length of result
        in_frequency:
        out_frequency:

    Return: Dict of heart rate time series and its sample frequency
    with keys 'sample_frequency' and 'hr'.

    """

    # Create periods, an array of periods that is uniformly sampled at
    # out_frequency
    n_out = int((out_frequency * time_out).to('').magnitude)
    n_in = len(time_period)
    # Number of samples of inverse heart rate
    out_periods = numpy.empty(n_out)

    index_in = 0
    next_time = 0
    next_period = 0
    last_time = 0
    last_period = 0
    dp_dt = 0

    def next_good_period():
        """Find the next good interval

        Update values of index_in, last_time, next_time, last_period,
        next_period

        """
        nonlocal last_time, next_time, last_period, next_period, index_in, dp_dt
        for try_in in range(index_in, n_in):
            time, period = time_period[try_in]
            if time not in bad_intervals:
                last_time = next_time
                last_period = next_period
                next_time = time
                next_period = period
                index_in = try_in + 1
                dp_dt = (next_period - last_period) / (next_time - last_time)
                return
        index_in = n_in
        last_time = next_time
        last_period = next_period
        dp_dt = 0

    next_good_period()
    assert index_in < n_in

    # Assign first out_periods
    for k_out in range(n_out):
        if k_out / out_frequency >= next_time / in_frequency:
            k_start = k_out
            break
        out_periods[k_out] = next_period

    # Make values of last_* and next_* valid
    next_good_period()
    assert index_in < n_in

    # Assign a value to periods[k_out] for every k_out
    for k_out in range(k_start, n_out):
        if k_out / out_frequency >= next_time / in_frequency:
            next_good_period()
            if index_in >= n_in:
                for k in range(k_out, n_out):
                    out_periods[k] = last_period
                break
        t_k = k_out * in_frequency / out_frequency
        out_periods[k_out] = last_period + (t_k - last_time) * dp_dt

    # Return results in beats per minute
    out_periods /= in_frequency
    return {
        'sample_frequency': out_frequency,
        'hr': (1.0 / out_periods).to('1/minute')
    }


def main(argv=None):
    """Derive periodic time series of heart rate estimates from state sequences
    """

    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    with open(args.states_file, 'rb') as _file:
        states = pickle.load(_file)
    with open(args.likelihood, 'rb') as _file:
        likelihood = pickle.load(_file)

    input_sample_frequency = 100 * PINT('Hz')
    time_out = len(states) / input_sample_frequency

    time_period, bad_intervals = find_beats_and_periods(
        states, likelihood=likelihood, censor_fraction=args.censor)
    hr_dict = calculate(time_period, bad_intervals, time_out)
    with open(args.heart_rate_file, 'wb') as _file:
        pickle.dump(hr_dict, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
