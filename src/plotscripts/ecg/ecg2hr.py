""" ecg2hr.py Illustrate ecg -> state sequence -> heart rate

python ecg2hr.py ../../../build/derived_data/ECG/a01_self_AR3/ \
../../../build/derived_data/apnea/ecgs/a01 foo.pdf

"""
import sys
import argparse
import pickle
import os

import numpy
import pint

import plotscripts.utilities

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Plot ECG and decoded states')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--t_start',
                        type=float,
                        default=352.07,
                        help="Time in minutes")
    parser.add_argument('--t_stop',
                        type=float,
                        default=352.2,
                        help="Time in minutes")
    parser.add_argument('hr_dir',
                        type=str,
                        help='Path dir of states and heart rate')
    parser.add_argument('ecg_path', type=str, help='Path to ecg data')
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Make time series picture with ecg and state data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)
    t_start = args.t_start * PINT('minutes')
    t_stop = args.t_stop * PINT('minutes')

    # Read ECGs
    with open(args.ecg_path, 'rb') as _file:
        _dict = pickle.load(_file)
    ecg = _dict['ecg']
    ecg_times = _dict['times'] * PINT('seconds')

    # Read states
    states_path = os.path.join(args.hr_dir, 'states')
    with open(states_path, 'rb') as _file:
        states = pickle.load(_file)
    state_times = numpy.arange(0, len(states)) / (100 * 60) * PINT('minutes')

    # Read heart rate
    heart_rate_path = os.path.join(args.hr_dir, 'heart_rate')
    with open(heart_rate_path, 'rb') as _file:
        pickle_dict = pickle.load(_file)
    hr = pickle_dict['hr'].to('1/minute').magnitude
    print(f"{pickle_dict['sample_frequency']=}")
    hr_times = numpy.arange(len(hr)) / pickle_dict['sample_frequency']

    fig, (ecg_axes, state_axes, hr_axes) = pyplot.subplots(nrows=3,
                                                           ncols=1,
                                                           sharex=True,
                                                           figsize=(6, 10))

    # States in middle plot
    n_start, n_stop = numpy.searchsorted(
        state_times.to('minutes').magnitude,
        (t_start.to('minutes').magnitude, t_stop.to('minutes').magnitude))
    times = ecg_times[n_start:n_stop].to('minutes').magnitude
    state_interval = states[n_start:n_stop]
    state_axes.plot(times, state_interval)

    indices = numpy.nonzero((state_interval == 31))[0]
    state_axes.plot(times[indices],
                    state_interval[indices],
                    marker='x',
                    color='red',
                    linestyle='',
                    markersize=5)
    state_axes.set_ylabel(r'State')

    # ECG in upper plot.  (Times for ECG are the same as times for states.)
    ecg_interval = ecg[n_start:n_stop]
    ecg_axes.plot(times, ecg_interval, label='ECG')

    ecg_axes.plot(times[indices],
                  ecg_interval[indices],
                  marker='x',
                  color='red',
                  linestyle='',
                  markersize=5)
    ecg_axes.set_ylabel(r'ECG $/$ mv')

    # Heart rate in lower plot.
    n_start, n_stop = numpy.searchsorted(
        hr_times.to('minutes').magnitude,
        (t_start.to('minutes').magnitude, t_stop.to('minutes').magnitude))
    times = hr_times[n_start:n_stop].to('minutes').magnitude
    hr_axes.plot(times, hr[n_start:n_stop])
    hr_axes.plot(times, hr[n_start:n_stop], marker='.', color='black', linestyle='', markersize=8)
    hr_axes.set_ylabel(r'Heart Rate $\times$ minute')
    hr_axes.set_xlabel(r'Time $/$ minute')

    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
