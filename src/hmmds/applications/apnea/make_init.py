"""make_init.py Use parameters in first argument to craft call to
model_init.py

EG: $ python make_init.py power0.9threshold-12ar5prom6.1 \
norm_power0.9threshold-12ar5prom6.1_masked

This script calls make in a subprocess to make config6.1.pkl, and then
calls python in a second subprocess to run model_init.py to make
norm_power0.9threshold-12ar5prom6.1_masked

I wrote this ugly hack to use in a Makefile because make doesn't
support multiple free parameters in a rule.

"""

import sys
import argparse
import subprocess


def parse_args(argv):
    """ A single line argument
    """

    parser = argparse.ArgumentParser("Map model name to parameter argunments")
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
def two_intervals(args):
    d = parse_pattern(args.pattern, 'power threshold ar prom')
    make_config = f"make config{d['prom']}.pkl"
    run_model_init = f"""
      python model_init.py
      --power 1 1 {d['power']} 1
      --threshold 1.0e{d['threshold']}
      --AR_order {d['ar']}
      config{d['prom']}.pkl two_intervals {args.out}"""
    return make_config, run_model_init


@register
def two_normalized(args):
    d = parse_pattern(args.pattern, 'power threshold ar prom')
    make_config = f"make norm_config{d['prom']}.pkl"
    run_model_init = f"""
      python model_init.py
      --power 1 1 {d['power']} 1
      --threshold 1.0e{d['threshold']}
      --AR_order {d['ar']}
      norm_config{d['prom']}.pkl two_normalized {args.out}"""
    return make_config, run_model_init


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
