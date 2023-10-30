"""make_init.py Use parameters in first argument to craft call to
model_init.py

EG: $ python make_init.py power0.9threshold-12ar5prom6.1 \
norm_power0.9threshold-12ar5prom6.1_masked

calls make in a subprocess to make config6.1.pkl, and then calls
python in a second subprocess to run model_init.py to make
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

    parser.add_argument('pattern',
                        type=str,
                        help='eg, power1.5threshold-12ar5prom6.1')
    parser.add_argument('out', type=str, help='path for writing initial model')
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    """Create an hmm and write it as a pickle.
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    pattern = args.pattern
    keys = 'power threshold ar prom'.split()
    starts = list(pattern.find(key) + len(key) for key in keys)
    ends = list(pattern.find(key) for key in keys[1:])
    ends.append(len(pattern))
    numbers = list(pattern[start:end] for start, end in zip(starts, ends))
    d = dict((key, number) for key, number in zip(keys, numbers))
    run_model_init = f"python model_init.py --power_and_threshold {d['power']} 1.0e{d['threshold']} --AR_order {d['ar']} norm_config{d['prom']}.pkl two_normalized {args.out}"
    make_config = f"make norm_config{d['prom']}.pkl"
    subprocess.run(make_config.split(), check=True)
    subprocess.run(run_model_init.split(), check=True)
    print(run_model_init)


if __name__ == "__main__":
    sys.exit(main())
