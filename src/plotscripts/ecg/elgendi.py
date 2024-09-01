"""elgendi.py Makes figure for ds23.pdf comparing hmm and Elgendi QRS detectors

elgendi.py constant_a03.pdf

"""
import sys
import argparse

import numpy
import pint

import plotscripts.utilities
from hmmds.applications.apnea.ECG import utilities

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Plot ECG and decoded states')
    utilities.common_arguments(parser)
    parser.add_argument(
        '--before_after_slow',
        nargs=3,
        type=int,
        default=(18, 30, 3),
        help=
        "Number of transient states before and after R in ECG, and number of slow states."
    )
    parser.add_argument('--tag_ecg',
                        action='store_false',
                        help="Invoke tagging in utilities.read_ecgs()")
    parser.add_argument('--records', type=str, nargs='+', default=['a03'])
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('fig_path', type=str, help="path to figure")
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def main(argv=None):
    """Make time series picture with ecg and state data.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    states = utilities.read_states(args, 'a03')
    # FixMe: Need rtimes from elgendi
    # rtimes = utilities.read_rtimes(args, 'a03')[0]  # * PINT('seconds')
    joint_segment = utilities.read_ecgs(args)[0]
    ecg = joint_segment['ecg']
    ecg_times = numpy.arange(len(ecg)) / (100 * PINT('Hz'))
    fig, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(6, 4))

    n_start, n_stop = numpy.searchsorted(
        ecg_times.to('minutes').magnitude, (59.5, 59.7))
    minutes = ecg_times[n_start:n_stop].to('minutes').magnitude
    ecg_segment = ecg[n_start:n_stop]
    states_segment = states[n_start:n_stop]
    axes.plot(minutes, ecg_segment)

    # Find places where state is 32
    indices = numpy.nonzero(states_segment == 32)[0]
    axes.plot(minutes[indices],
              ecg_segment[indices],
              marker='x',
              color='red',
              linestyle='',
              markersize=15,
              label='hmm')

    # seconds = ecg_times[n_start:n_stop].to('seconds').magnitude
    # foo, bar = numpy.searchsorted(rtimes, (seconds[0], seconds[-1]))
    # indices = numpy.searchsorted(seconds, rtimes[foo:bar])
    # axes.plot(minutes[indices],
    #           ecg_segment[indices],
    #           marker='*',
    #           color='black',
    #           linestyle='',
    #           markersize=15,
    #           label='py-ecg')

    axes.set_xlabel(r'$t$/minutes')
    axes.set_ylabel(r'$a03$ ecg/mV')
    axes.legend()
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
