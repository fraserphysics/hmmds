"""states2hr.py: Map a decoded state sequence to a heart rate time
series in a format comparable to rtimes2hr.py.

python states2hr.py states_file heart_rate_file

Read a states_file, estimate a heart rate time series sampled at the
frequency --samples_per_minute and write the result.

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


def calculate(states: numpy.ndarray, r_state=20, outlier_state=0) -> dict:
    """ Calculate heart rate and low pass heart rate

    Args:
        states: Decoded state sequence sampled at 100Hz
        r_state: The state that corresponds to the R peak
        outlier_state: The state for ECG samples that are implausible

    Return: 
    """

    in_frequency = 100 * PINT('Hz')
    out_frequency = 2 * PINT('Hz')

    time_period_index = []
    r_indices = numpy.nonzero(states == r_state)[0]
    for next_index, r_index in zip(r_indices[1:], r_indices[:-1]):
        if numpy.count_nonzero(
                states[r_index:next_index] == outlier_state) == 0:
            time_period_index.append([r_index, next_index - r_index])
    time_period = numpy.array(time_period_index) / in_frequency

    # Create periods, an array of periods that is uniformly sampled at
    # out_frequency

    last_time = (len(states) / 100) * PINT('seconds')
    n_out = int((out_frequency * last_time).to('').magnitude)
    # Number of samples of inverse heart rate
    periods = numpy.ones(n_out) * PINT('s')
    if len(time_period_index) < 3:
        # Worthless data in states
        return {
            'sample_frequency': out_frequency,
            'hr': (1 / periods).to('1/minute')
        }

    for k_out in range(n_out):
        periods[k_out] = time_period[0, 1]
        if k_out / out_frequency >= time_period[0, 0]:
            k_start = k_out
            break
    for k_out in range(n_out - 1, -1, -1):
        periods[k_out] = time_period[-1, 1]
        if k_out / out_frequency <= time_period[-1, 0]:
            k_stop = k_out
            break

    # t_old        t_new
    #   |            |
    #------------------------------------------
    #        |
    #       t_k

    i_time_period = 1

    # Assign a value to periods[k_out] for every k_out
    for k_out in range(k_start, k_stop):
        t_k = k_out / out_frequency

        # Move t_old and t_new forward till t_k is in range
        while t_k > time_period[i_time_period, 0]:
            i_time_period += 1
            if i_time_period >= len(time_period) - 1:
                break
        t_old, period_old = time_period[i_time_period - 1]
        t_new, period_new = time_period[i_time_period]

        # Calculate slope of period between t_old and t_new
        drdt = (period_new - period_old) / (t_new - t_old)
        periods[k_out] = period_old + (t_k - t_old) * drdt

    # Return results in beats per minute
    return {
        'sample_frequency': out_frequency,
        'hr': (1.0 / periods).to('1/minute')
    }


def main(argv=None):
    """Derive periodic time series of heart rate estimates from state sequences
    """

    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    with open(args.states_file, 'rb') as _file:
        states = pickle.load(_file)
    hr_dict = calculate(states, args.r_state)
    with open(args.heart_rate_file, 'wb') as _file:
        pickle.dump(hr_dict, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
