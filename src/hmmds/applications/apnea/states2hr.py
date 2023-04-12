"""states2hr.py: Map a decoded state sequence to a heart rate time
series in a format comparable to rtimes2hr.py.

python states2hr.py states_dir heart_rate_dir a01 b01 c01 x01 etc

For each record R listed, read states_dir/R, estimate a heart rate
time series sampled at the frequency --samples_per_minute and write
the result to a lphr_dir/R, eg,

python states2hr.py --samples_per_minute 10 --r_state 20 model_dir/states model_dir/heart_rate

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
    parser.add_argument('--r_state', type=int, default=20, help='state index for R peak')
    parser.add_argument('--outlier_state', type=int, default=0, help='state index for outliers')
    parser.add_argument('states_dir', type=str, help='Path to decoded states')
    parser.add_argument('lphr_dir',
                        type=str,
                        help='Path to low pass heart rate data')
    parser.add_argument('record_names',
                        type=str,
                        help='EG, a01 a02 b01 b02 x01',
                        nargs='+')
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
        if numpy.count_nonzero(states[r_index:next_index] == outlier_state) == 0:
            time_period_index.append([r_index, next_index-r_index])
    time_period = numpy.array(time_period_index) / in_frequency

    # Create periods, an array of periods that is uniformly sampled at
    # out_frequency

    last_time = (len(states) / 100) * PINT('seconds')
    n_out = int((out_frequency * last_time).to('').magnitude)
    # Number of samples of inverse heart rate
    periods = numpy.empty(n_out) * PINT('s')

    for k_out in range(n_out):
        periods[k_out] = time_period[0,1]
        if k_out/out_frequency >= time_period[0,0]:
            k_start = k_out
            break
    for k_out in range(n_out-1,-1,-1):
        periods[k_out] = time_period[-1,1]
        if k_out/out_frequency <= time_period[-1,0]:
            k_stop = k_out
            break

    # t_old        t_new
    #   |            |
    #------------------------------------------
    #        |
    #       t_k

    i_time_period = 1

    # Assign a value to periods[k_out] for every k_out
    for k_out in range(k_start,k_stop):
        t_k = k_out / out_frequency

        # Move t_old and t_new forward till t_k is in range
        while t_k > time_period[i_time_period,0]:
            i_time_period += 1
            if i_time_period >= len(time_period)-1:
                break
        t_old, period_old = time_period[i_time_period-1]
        t_new, period_new = time_period[i_time_period]

        # Calculate slope of period between t_old and t_new
        drdt = (period_new - period_old) / (t_new - t_old)
        periods[k_out] = period_old + (t_k - t_old) * drdt

    # Transform periods to heart rate
    heart_rate = 1.0 / periods
    # Write results in beats per minute
    result = {
        'sample_frequency': out_frequency,
        'hr': heart_rate.to('1/minute')
    }
    return result


def main(argv=None):
    """Derive periodic time series of heart rate estimates from state sequences
    """

    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    for name in args.record_names:
        with open(os.path.join(args.states_dir, name), 'rb') as _file:
            states = pickle.load(_file)
        hr_dict = calculate(states)
        with open(os.path.join(args.lphr_dir, name), 'wb') as _file:
            pickle.dump(hr_dict, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
