"""particle.py Apply particle filter to quantized Lorenz data and
write two files to a directory

EG: python particle.py result_dir

or

python wrapper_particle.py $(Args) study_dir

Called as wrapper_particle.py, this code would a subdirectory in
study_dir with a complex name that reflects the arguments.

In result_dir, write states_boxes and weights at each time to one file
and a dict to a pickle file with the following keys:

args   Parsed command line arguments and defaults
x_all  Long (args.n_initialize) Lorenz trajectory
y_q    Quantized data derived from x_all[:args.n_y]
gamma  Incremental likelihood from p_filter(y_q, ...)
bins   Bin boundaries
log_dict Record of progress of function named forward

"""

from __future__ import annotations
import sys
import argparse
import pickle
import os

import numpy
import numpy.linalg

from hmmds.synthetic.bounds import lorenz
from hmmds.synthetic.bounds import benettin
from hmmds.synthetic.bounds.filter import Filter


def parse_args(argv):
    """Parse a command line.
    """
    parser = argparse.ArgumentParser(
        description='Apply particle filter to Lorenz data')
    parser.add_argument(
        '--r_threshold',
        type=float,
        default=0.001,
        help='Maximum ratio of quadratic to linear edge velocity')
    parser.add_argument('--r_extra',
                        type=float,
                        default=2.0,
                        help='Extra factor for dividing boxes')
    parser.add_argument('--edge_max',
                        type=float,
                        default=0.2,
                        help='Divide edges bigger than this')
    parser.add_argument('--margin',
                        type=float,
                        default=.5,
                        help='Keep outside particles this close to boundaries')
    parser.add_argument('--s_augment',
                        type=float,
                        default=.0005,
                        help='Grow boxes at each step')
    parser.add_argument('--n_y',
                        type=int,
                        default=1000,
                        help='Number of test observations')
    parser.add_argument('--n_initialize',
                        type=int,
                        default=15000,
                        help='Number simulated points to cover attractor')
    parser.add_argument('--n_quantized',
                        type=int,
                        default=4,
                        help='Cardinality of test observations')
    parser.add_argument('--time_step', type=float, default=0.15)
    parser.add_argument('--t_relax',
                        type=float,
                        default=50.0,
                        help='Time to move to attractor')
    parser.add_argument('--atol',
                        type=float,
                        default=1e-8,
                        help='Absolute error tolerance for integrator')
    parser.add_argument(
        '--resample',
        type=int,
        nargs=2,
        default=(10000, 4000),
        help='If more than resample[0] particles, resample to resample[1]')
    parser.add_argument('--random_seed',
                        type=int,
                        default=7,
                        help='For random number generator')
    parser.add_argument('result_dir',
                        type=str,
                        help='write results to this path')
    return parser.parse_args(argv)


def make_data(args):
    """Generate quantized data y_q and bins for filter.  Also return
    vector data x_all for debugging.  Don't use lorenz.n_steps because
    results from n_steps and integrate_tangent diverge from each other
    too fast for filtering with integrate_tangent on data from n_steps
    to work.  Instead, generate data by imitating the integration in
    call stack Filter.forward -> Filter.forecast_x -> Particle.step.

    """

    # Relax to a point near the attractor
    x_0 = benettin.relax(args, numpy.ones(3))[0]

    tangent = numpy.eye(3) * 0.1
    x_all = numpy.empty((args.n_y + args.n_initialize, 3))
    x_all[0, :] = x_0
    for i in range(1, args.n_y + args.n_initialize):
        x_all[i, :], _ = lorenz.integrate_tangent(args.time_step,
                                                  x_all[i - 1, :],
                                                  tangent,
                                                  atol=args.atol)
    assert x_all.shape == (args.n_y + args.n_initialize, 3)
    bins = numpy.linspace(-20, 20, args.n_quantized + 1)[1:-1]
    y_q = numpy.digitize(x_all[:, 0], bins)
    return y_q, x_all, bins


def initialize(p_filter: Filter,
               y_data: numpy.ndarray,
               y_reference: numpy.ndarray,
               x_reference: numpy.ndarray,
               x_0=None):
    """Create a particle filter (Filter instance)

    Args:
        p_filter:
        y_data: Simulated observations
        y_reference: Other simulated observations for matching
        x_reference: Simulated state sequence
        x_0: Initial true state that generated y_data

    If an x_0 vector is not passed, one is estimated by searching
    y_reference for a match to the first part of y_data.

    """

    def find_best(y_data: numpy.ndarray, y_reference: numpy.ndarray):
        """Find start and length of longest sequence in y_reference
        that matches the beginning of y_data.

        Args:
            y_data: Sequence of simulated integer observations
            y_reference: Different sequence of simulated integer observations
        """
        best = (0, 0)  # (start, length)
        counters: dict[int, int] = {}
        for n, y in enumerate(y_reference):
            for start, count in list(counters.items()):
                if y_data[count] == y:
                    counters[start] += 1
                    if counters[start] > best[1]:
                        best = (start, counters[start])
                    if counters[start] == len(y_data):  # Match of all y_data
                        return best
                else:
                    del counters[start]
            if y == y_data[0]:
                counters[n] = 1
        return best

    if x_0 is None:
        x_0 = x_reference[find_best(y_data, y_reference)[0]]

    delta = 0.2  # Size of initial boxes
    p_filter.initialize(x_0, delta)


def particle(args):
    """Run particle filter on quantized Lorenz data, and write
    results to two files in the directory _args.result_dir_.

    """
    npy_file = open(os.path.join(args.result_dir, 'states_boxes.npy'), 'wb')
    assert args.n_y % 5 == 0

    y_all, x_all, bins = make_data(args)
    y_q = y_all[:args.n_y]
    y_reference = y_all[args.n_y:]
    rng = numpy.random.default_rng(args.random_seed)
    p_filter = Filter(args, bins, rng)
    initialize(p_filter, y_q, y_reference, x_all[args.n_y:], x_all[0])
    gamma = numpy.ones(len(y_q))

    log_dict = {}
    p_filter.forward(y_q, (0, len(y_q)), gamma, npy_file=npy_file, log=log_dict)

    # Write results
    result_dict = {
        'args': args,
        'gamma': gamma,  # (100,) From x_all[:100]
        'y_q': y_q,  # (100,) From x_all[:100]
        'bins': bins,  # (3,)  
        'x_all': x_all,  # (15100, 3)
        'log_dict': log_dict,
    }

    with open(os.path.join(args.result_dir, 'dict.pkl'), 'wb') as _file:
        pickle.dump(result_dict, _file)
    return 0


def wrapper(args):
    """Create directory with name derived from args and write results there
    """
    # Create directory name
    args_dict = vars(args).copy()
    del args_dict['result_dir']
    keys = list(args_dict.keys())
    keys.sort()
    name_list = []
    for key in keys:
        name_list.append(f'{key}..{args_dict[key]}..')
    name = ''.join(name_list).replace(', ', '..').replace(']',
                                                          '').replace('[', '')
    os.makedirs(args.result_dir)
    return particle(args)


def main():
    """Run particle filter on Lorenz data

    """
    args = parse_args(sys.argv[1:])
    if sys.argv[0] == 'particle.py':
        return particle(args)
    if sys.argv[0] == 'wrapper_particle.py':
        return wrapper(args)


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
