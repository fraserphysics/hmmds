"""score.py make a file that looks like this:

Name  Apnea   Normal  Apnea->Normal   Normal->Apnea   Total   Error
a01     420       60       42  0.10         6  0.10     480    0.10
.
.
.
c10      60      420        6  0.10        42  0.10     480    0.10

Sum   10000    10000     1000  0.10      1000  0.10   20000    0.10

From Makefile:
python score.py --root ${ROOT} ${DATA}/pass2_report ${EXPERT} $@ $(As) $(Bs) $(Cs)
"""
import sys
import os
import argparse
import glob

import numpy

import hmmds.applications.apnea.utilities

format = '{0:5s} {1:-6d}    {2:-5d} {3:-5d}     {4:-3.2f}   {5:-5d}    {6:-3.2f}   {7:-5d}    {8:-3.2f}'


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        "Compare classification of minutes by HMM and expert")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument('result',
                        type=str,
                        help='Write result to this path',
                        default='score_report')
    parser.add_argument('result',
                        type=str,
                        help='Write result to this path',
                        default='score_report')
    parser.add_argument('names',
                        type=str,
                        nargs='*',
                        help='names of records to analyze')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def analyze(name: str, _expert: numpy.ndarray, _pass2: numpy.ndarray,
            report) -> list:
    """Compare expert and pass2 and write a single line to report

    Args:
        name: Eg, 'a01'
        expert: array with expert[t] = 0 for normal, and expert[t] = 1 for apnea
        pass2: array with same format as expert
        report: A file object open for writing
    """
    n_expert = len(_expert)
    n_pass2 = len(_pass2)
    n_min = min(n_expert, n_pass2)
    if n_expert > n_pass2:
        print('For {0} n_expert={1} n_pass2={2}'.format(name, n_expert,
                                                        n_pass2))

    # Truncate data to length of minimum
    expert = _expert[:n_min]
    pass2 = _pass2[:n_min]

    n_apnea = expert.sum()
    n_normal = n_min - n_apnea
    apnea2apnea = (expert & pass2)
    n_apnea2normal = n_apnea - apnea2apnea.sum()
    errors = (expert ^ pass2)  # Exclusive or
    n_normal2apnea = errors.sum() - n_apnea2normal

    def safe(numerator, divisior):
        if numerator == 0:
            return 0
        return numerator / divisior

    a2n_fraction = safe(n_apnea2normal, n_apnea)
    n2a_fraction = safe(n_normal2apnea, n_normal)
    error_fraction = (errors.sum() / n_min)
    values = (name, n_apnea, n_normal, n_apnea2normal, a2n_fraction,
              n_normal2apnea, n2a_fraction, n_min, error_fraction)
    print(format.format(*values), file=report)
    return values


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    def get_names(letter):
        return [
            os.path.basename(x)
            for x in glob.glob('{0}/{1}*'.format(args.heart_rate_dir, letter))
        ]

    if not args.names:
        args.names = get_names('a') + get_names('b') + get_names(
            'c') + get_names('x')

    n_apnea = 0
    n_normal = 0
    n_a2n = 0
    n_n2a = 0
    n_total = 0

    with open(args.result, 'w') as report:
        print(
            'Name   Apnea   Normal  Apnea->Normal   Normal->Apnea   Total   Error',
            file=report)
        for name in args.names:
            expert = hmmds.applications.apnea.utilities.read_expert(
                args.expert, name)
            pass2 = hmmds.applications.apnea.utilities.read_expert(
                args.pass2, name)
            values = analyze(name, expert, pass2, report)
            _, apnea, normal, a2n, _, n2a, _, total, _ = values
            n_apnea += apnea
            n_normal += normal
            n_a2n += a2n
            n_n2a += n2a
            n_total += total

        error_fraction = (n_a2n + n_n2a) / n_total
        a2n_fraction = n_a2n / n_apnea
        n2a_fraction = n_n2a / n_normal
        values = ('Total', n_apnea, n_normal, n_a2n, a2n_fraction, n_n2a,
                  n2a_fraction, n_total, error_fraction)
        print(format.format(*values), file=report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
