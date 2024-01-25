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
def two_intervals(args):
    """ Observation components slow peak interval class
    """
    d = parse_pattern(args.pattern, 'power threshold ar prom')
    make_config = f"make config{d['prom']}.pkl"
    run_model_init = f"""
      python model_init.py
      --power_dict slow 1 peak 1 interval {d['power']} class 1
      --threshold 1.0e{d['threshold']}
      --AR_order {d['ar']}
      config{d['prom']}.pkl two_intervals {args.out}"""
    return make_config, run_model_init


@register
def two_normalized(args):
    """ Observation components slow peak interval class
    """
    d = parse_pattern(args.pattern, 'power threshold ar prom')
    make_config = f"make norm_config{d['prom']}.pkl"
    run_model_init = f"""
      python model_init.py
      --power_dict slow 1 peak 1 interval {d['power']} class 1
      --threshold 1.0e{d['threshold']}
      --AR_order {d['ar']}
      norm_config{d['prom']}.pkl two_normalized {args.out}"""
    return make_config, run_model_init


@register
def varg2state(args):
    """
    ar AR_order
    fs model_sample_frequency  samples per minute
    lpp low_pass_period        seconds
    rc band_pass_center        cycles per minute
    rw band_pass_width         cycles per minute
    rs respiration_smooth      cycles per minute
    
    """
    d = parse_pattern(args.pattern, 'ar fs lpp rc rw rs')
    make_config = f"make norm_config4.pkl"
    run_model_init = f"""
      python model_init.py
      --AR_order {d['ar']}
      --model_sample_frequency {d['fs']}
      --low_pass_period {d['lpp']}
      --band_pass_center {d['rc']}
      --band_pass_width {d['rw']}
      --respiration_smooth {d['rs']}
      norm_config4.pkl varg2state {args.out}"""
    return make_config, run_model_init


@register
def four_state(args):
    """
    ar AR_order
    fs model_sample_frequency  samples per minute
    lpp low_pass_period        seconds
    rc band_pass_center        cycles per minute
    rw band_pass_width         cycles per minute
    rs respiration_smooth      cycles per minute
    pt Prominence threshold
    vp power for varg component
    ip power for interval component

    Observation components: hr_respiration interval class
    """
    d = parse_pattern(args.pattern, 'ar fs lpp rc rw rs pt vp ip')
    make_config = f"make norm_config{d['pt']}.pkl"
    run_model_init = f"""
      python model_init.py
      --AR_order {d['ar']}
      --model_sample_frequency {d['fs']}
      --low_pass_period {d['lpp']}
      --band_pass_center {d['rc']}
      --band_pass_width {d['rw']}
      --respiration_smooth {d['rs']}
      --power_dict hr_respiration {d['vp']} interval {d['ip']} class 1 --
      norm_config{d['pt']}.pkl four_state {args.out}"""
    return make_config, run_model_init


@register
def varg2chain(args):
    """ Variables:
            pt: Prominence threshold
            ldt: Log detection threshold
            vp: Varg Power: Exponential weight of component
            ip: Interval Power: Exponential weight of component

    Observation components: hr_respiration peak interval class
    """
    d = parse_pattern(args.pattern, 'pt ldt vp ip')
    threshold = 10.0**float(d['ldt'])
    low_pass_period = int(args.low_pass_period.to('second').magnitude)
    sample_frequency = int(args.model_sample_frequency.to('1/minute').magnitude)
    band_pass_center = args.band_pass_center.to('1/minute').magnitude
    band_pass_width = args.band_pass_width.to('1/minute').magnitude
    make_config = f"make norm_config{d['pt']}.pkl"
    run_model_init = f"""
      python model_init.py
      --power_dict hr_respiration {d['vp']} peak 1 interval {d['ip']} class 1
      --threshold {threshold}
      --AR_order {args.AR_order}
      --model_sample_frequency {sample_frequency}
      --low_pass_period {low_pass_period}
      --band_pass_center {band_pass_center}
      --band_pass_width {band_pass_width} 
      norm_config{d['pt']}.pkl varg2chain {args.out}"""
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
