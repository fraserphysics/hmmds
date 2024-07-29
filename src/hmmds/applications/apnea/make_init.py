"""make_init.py Use parameters in first argument to craft call to
model_init.py

EG: $ python make_init.py FixMe

This script calls make in a subprocess to run model_init.py to make
FixMe

I wrote this ugly hack to use in a Makefile because make doesn't
support multiple free parameters in a rule.

"""

import sys
import argparse
import subprocess

import utilities


def parse_args(argv):
    """ A single line argument
    """

    parser = argparse.ArgumentParser("Map model name to parameter argunments")
    utilities.common_arguments(parser)
    parser.add_argument('--debug',
                        action='store_true',
                        help="Print issued commands to stdout")
    parser.add_argument('key',
                        type=str,
                        help='Determines type of model, eg, two_normalized')
    parser.add_argument(
        'pattern',
        type=str,
        help='Sets values of parameters, eg, power1.5threshold-12ar5prom6.1')
    parser.add_argument('out', type=str, help='path for writing initial model')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


MODELS = {}  # This dict and the register decorator mimic those in
# model_init.py so that the same keys used there can also
# be used here.


def register(func):
    """Decorator that puts function in MODELS dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    MODELS[func.__name__] = func
    return func


@register
def multi_state(args):
    """
    ar AR_order
    fs model_sample_frequency  samples per minute
    lpp low_pass_period        seconds
    rc band_pass_center        cycles per minute
    rw band_pass_width         cycles per minute
    rs respiration_smooth      cycles per minute

    Observation components: hr_respiration class
    """
    d = parse_pattern(args.pattern, 'ar fs lpp rc rw rs')
    run_model_init = f"""
      python model_init.py
      --AR_order {d['ar']}
      --model_sample_frequency {d['fs']}
      --low_pass_period {d['lpp']}
      --band_pass_center {d['rc']}
      --band_pass_width {d['rw']}
      --respiration_smooth {d['rs']}
    multi_state {args.out}"""
    return (run_model_init,)


def parse_pattern(pattern, key_string):
    """ Make a dict from pairs keynumber in pattern
    """

    keys = key_string.split()
    starts = list(pattern.find(key) + len(key) for key in keys)
    ends = list(pattern.find(key) for key in keys[1:])
    ends.append(len(pattern))
    numbers = list(pattern[start:end] for start, end in zip(starts, ends))
    return dict((key, number) for key, number in zip(keys, numbers))


def main(argv=None):
    """Create an hmm and write it as a pickle.
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    for call_string in MODELS[args.key](args):
        if args.debug:
            print(f'''
from make_init.py issue:
{call_string}
''')
        subprocess.run(call_string.split(), check=True)


if __name__ == "__main__":
    sys.exit(main())
