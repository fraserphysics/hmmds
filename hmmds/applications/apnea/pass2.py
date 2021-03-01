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


def analyze(name, model, common, report):
    """

    Args:
        name: Eg, 'a01'
        model: An HMM with an observation model for both heart rate and respiration that supports decoding a sequence of groups
        common: Globally shared paths and parameters for the apnea project

    Returns:
        A string that imitates the expert file
    """

    data = hmmds.applications.apnea.utilities.heart_rate_respiration_data(
        name, common)
    print('decoding {0}'.format(name))
    class_sequence = (numpy.array(model.broken_decode([data])) -
                      0.5) * 2  # +/- 1
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
    return


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser("Create and write pass2_report")
    parser.add_argument('--root',
                        type=str,
                        default='../../../',
                        help='Root directory of project')
    parser.add_argument('--names',
                        type=str,
                        nargs='*',
                        help='names of records to analyze')
    parser.add_argument('result',
                        type=str,
                        help='Write result to this path',
                        default='pass2_report')
    args = parser.parse_args(argv)
    common = hmmds.applications.apnea.utilities.Common(args.root)

    with open(common.pass1 + '.pickle', 'rb') as _file:
        report_list = pickle.load(_file)
    pass1_dict = dict((x.name, x) for x in report_list)

    def get_names(letter):
        return [
            os.path.basename(x) for x in glob.glob('{0}/{1}*'.format(
                common.heart_rate_directory, letter))
        ]

    if not args.names:
        args.names = get_names('a') + get_names('b') + get_names(
            'c') + get_names('x')

    models = {}
    for name in 'Low Medium High'.split():
        with open(common.get('model' + name), 'rb') as _file:
            models[name] = pickle.load(_file)

    with open(args.result, 'w') as report:
        for name in args.names:
            pass1item = pass1_dict[name]
            model = models[pass1item.level]
            analyze(name, model, common, report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
