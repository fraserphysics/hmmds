"""pass2.py make a file that looks like this:

a01
 0 NNNNNNNNNNNNNAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 1 AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 2 AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 3 AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANNNNAAAAAAAAA
 4 AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 5 AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANNAAAAAA
 6 AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 7 AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
 8 AAAAAAAAA

a02
 0 NNNNNNNNNNNNNNNNNNNNNAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
.
.
.

"""
import sys
import os
import argparse
import glob
import pickle

import numpy

import hmmds.applications.apnea.utilities
import develop


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser("Create and write/pickle pass1_report")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('--names',
                        type=str,
                        nargs='*',
                        help='names of records to analyze')
    parser.add_argument('result',
                        type=str,
                        help='Write result to this path',
                        default='pass2_report')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def analyze(name, model, args, report):
    """Writes to the open file report a string that has the same form as
        the expert file

    Args:
        name: Eg, 'a01'
        model: An HMM with an observation model for both heart rate and respiration
            that supports decoding a sequence of groups
        args: Includes globally shared paths and parameters for the apnea project
        report: A file open for writing

    """

    print('decoding {0}'.format(name))
    data = hmmds.applications.apnea.utilities.heart_rate_respiration_data(
        name, args)
    # FixMe: Bundles are gone
    sequence = model.bundle_decode([data], fudge=0.7, power=1.0)
    #sequence = model.old_bundle_decode([data])

    if sequence is None:
        raise RuntimeError(
            'Failed to find any class sequence for {0}'.format(name))

    # The rest of the code writes the result in the same format as the
    # expert file
    class_sequence = (numpy.array(sequence) - 0.5) * 2  # +/- 1
    samples_per_minute = 10
    minutes_per_hour = 60
    samples_per_hour = minutes_per_hour * samples_per_minute
    n_samples = len(class_sequence)
    # -(- ...) to round up instead of down
    n_minutes = -(-n_samples // samples_per_minute)
    n_hours = -(-n_samples // samples_per_hour)
    print('{0}\n'.format(name), end='', file=report)
    for hour in range(n_hours):
        print(' {0:1d} '.format(hour), end='', file=report)
        minute_start = hour * minutes_per_hour
        minute_stop = min((hour + 1) * minutes_per_hour, n_minutes)
        for minute in range(minute_start, minute_stop):
            sample_start = minute * samples_per_minute
            sample_stop = min(n_samples, (minute + 1) * samples_per_minute)
            if class_sequence[sample_start:sample_stop].sum() > 0:
                print('A', end='', file=report)
            else:
                print('N', end='', file=report)
        print('\n', end='', file=report)


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    with open(args.pass1 + '.pickle', 'rb') as _file:
        report_list = pickle.load(_file)
    pass1_dict = dict((x.name, x) for x in report_list)

    def get_names(letter):
        return [
            os.path.basename(x) for x in glob.glob('{0}/{1}*'.format(
                args.heart_rate_directory, letter))
        ]

    if not args.names:
        args.names = get_names('a') + get_names('b') + get_names(
            'c') + get_names('x')

    # Build a dict by readings HMMs
    models = {}
    for name in 'model_Low model_Medium model_High'.split():
        with open(os.path.join(args.models_dir, name), 'rb') as _file:
            _, models[name] = pickle.load(_file)

    with open(args.result, 'w') as report:
        for name in args.names:
            pass1item = pass1_dict[name]
            # Choose the HMM based on the pass1 level
            model = models[pass1item.level]
            analyze(name, model, args, report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
