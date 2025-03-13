"""particle.py Apply variant of particle filter to quantized Lorenz data

Call with: python particle.py result_file

Create x_all, y_q and p_filter (initialized with x_all[0]).  Then run
filter on y_q.

Write a dict to a pickle file with the following keys:

x_all  Long (args.n_initialize) Lorenz trajectory
y_q    Quantized data derived from x_all[:args.n_y]
gamma  Incremental likelihood from p_filter(y_q, ...)
clouds Forecast and update particles at times args.clouds
bins   Bin boundaries.

"""

from __future__ import annotations
import sys
import argparse
import pickle

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
        '--clouds',
        type=int,
        nargs='*',
        help='each pair defines an interval in which to record the particles')
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
    parser.add_argument('result', type=str, help='write result to this path')
    return parser.parse_args(argv)


def make_marks(intervals: list, n_y: int) -> numpy.ndarray:
    """Create an array to determine which time steps are stored as clouds

    Args:
        intervals: From args.clouds
        n_y: Length of observation sequence

    """

    cloud_marks = numpy.zeros(n_y, dtype=bool)
    if intervals is None:
        return cloud_marks
    assert len(intervals) % 2 == 0
    for start, stop in zip(intervals, intervals[1:]):
        cloud_marks[start:stop] = True
    return cloud_marks


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


def main(argv=None):
    """Run particle filter on Lorenz data

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    assert args.n_y % 5 == 0
    cloud_marks = make_marks(args.clouds, args.n_y)

    y_all, x_all, bins = make_data(args)
    y_q = y_all[:args.n_y]
    y_reference = y_all[args.n_y:]
    rng = numpy.random.default_rng(args.random_seed)
    p_filter = Filter(args, bins, rng)
    initialize(p_filter, y_q, y_reference, x_all[args.n_y:], x_all[0])
    gamma = numpy.ones(len(y_q))
    clouds = {}  # keys are pairs (t,'forecast') or (t,'update')

    debug_times = set()
    # Run filter on y_q
    for t_start in range(0, len(y_q), 5):
        p_filter.forward(y_q, (t_start, t_start + 5), gamma, clouds)
        if not cloud_marks[t_start]:
            debug_times.add(t_start)
        if t_start - 25 in debug_times:
            debug_times.discard(t_start - 25)
            for t in range(t_start - 25, t_start - 20):
                del clouds[(t, 'forecast')]
                del clouds[(t, 'update')]
        if len(p_filter.particles) == 0:
            break

    # Write results
    result = {
        'gamma': gamma,  # (100,) From x_all[:100]
        'y_q': y_q,  # (100,) From x_all[:100]
        'bins': bins,  # (3,)  
        'x_all': x_all,  # (15100, 3)
        'clouds': clouds  # dict
    }

    with open(args.result, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
