"""profile_particle.py Profile code called by particle.py

Call with: python profile_particle.py profile_stats

Create x_all, y_q and p_filter (initialized with x_all[0]), profile
running filter on y_q, and print profile results.

"""

from __future__ import annotations
import sys
import cProfile
import pstats

import numpy

from hmmds.synthetic.bounds.filter import Filter
from hmmds.synthetic.bounds.particle import parse_args, make_data, initialize


def run_filter(args, y_all, x_all, bins):
    '''Profile the running of Filter.forward on y_all
    '''
    rng = numpy.random.default_rng(args.random_seed)
    p_filter = Filter(args, bins, rng)
    initialize(p_filter, y_all, y_all, x_all[args.n_y:], x_all[0])
    gamma = numpy.ones(len(y_all))
    cProfile.runctx('p_filter.forward(y_all, (0, len(y_all)), gamma)',
                    globals(), locals(), args.result)
    return gamma


def print_profile(args):
    '''Print functions with largest cumtime
    '''
    p = pstats.Stats(args.result)
    p.strip_dirs()
    p.sort_stats(pstats.SortKey.CUMULATIVE)
    p.print_stats(10)


def main(argv=None):
    """Profile the running of a particle filter on Lorenz data

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    args.n_initialize = 0
    args.margin = 0.01  # Default .5
    #args.s_augment = 5.0e-4  # Default 5.0e-4
    #args.resample = (100000, 40000)  # Default (10000, 4000)
    #args.r_threshold = 5e-6  # Default 0.001
    args.edge_max = 0.2  # Default .2
    args.n_y = 20  # Default 1000
    y_all, x_all, bins = make_data(args)
    run_filter(args, y_all, x_all, bins)
    print_profile(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
