"""particle.py Apply variant of particle filter to quantized Lorenz data

Call with: python particle.py result_file

"""

from __future__ import annotations
import sys
import argparse
import pickle

import numpy
import numpy.linalg

import hmmds.synthetic.bounds.lorenz as lorenz
import hmmds.synthetic.bounds.benettin as benettin


def parse_args(argv):
    """Parse a command line.
    """
    parser = argparse.ArgumentParser(
        description='Apply particle filter to Lorenz data')
    parser.add_argument('--epsilon_min',
                        type=float,
                        default=0.25,
                        help='Minimum length of box edges')
    parser.add_argument('--initial_dx',
                        type=float,
                        default=0.25,
                        help='Cell size for initialization')
    parser.add_argument(
        '--epsilon_ratio',
        type=float,
        default=5,
        help='Maximum length of box edges = ratio * epsilon_min')
    parser.add_argument('--n_y',
                        type=int,
                        default=1000,
                        help='Number of test observations')
    parser.add_argument('--n_quantized',
                        type=int,
                        default=4,
                        help='Cardinality of test observations')
    parser.add_argument('--time_step', type=float, default=0.05)
    parser.add_argument('--t_relax',
                        type=float,
                        default=10.0,
                        help='Time to move to attractor')
    parser.add_argument('--n_initialize',
                        type=int,
                        default=100000,
                        help='Number of time steps for initial particles')
    parser.add_argument('--atol',
                        type=float,
                        default=1e-7,
                        help='Absolute error tolerance for integrator')
    parser.add_argument('result', type=str, help='write result to this path')
    return parser.parse_args(argv)


def make_marks(n_y, cloud_intervals):
    """"""
    cloud_marks = numpy.zeros(n_y, dtype=bool)
    for start, stop in cloud_intervals:
        cloud_marks[start:stop] = True
    return cloud_marks


def main(argv=None):
    """Run particle filter on Lorenz data

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    assert args.n_y % 5 == 0
    # Relax to a point near the attractor
    relaxed_x = benettin.relax(args, numpy.ones(3))[0]

    # Generate quantized data y_q and x_0 for filter
    x_all = lorenz.n_steps(relaxed_x, args.n_y, args.time_step, atol=args.atol)
    assert x_all.shape == (args.n_y, 3)
    bins = numpy.linspace(-20, 20, args.n_quantized + 1)[1:-1]
    y_q = numpy.digitize(x_all[:, 0], bins)
    gamma = numpy.zeros(len(y_q))
    x_0 = x_all[-1]
    clouds = {}
    result = {
        'gamma': gamma,
        'x_all': x_all,
        'y_q': y_q,
        'clouds': clouds,
    }
    cloud_marks = make_marks( #
        len(y_q), #
        ( #
            (0,100), #
            ((int(len(y_q)*.99)), len(y_q)),
        )
    )

    # Initialize filter
    epsilon_max = args.initial_dx
    epsilon_min = epsilon_max / args.epsilon_ratio
    stretch = 1.25
    p_filter = benettin.Filter(epsilon_min, epsilon_max, bins, args.time_step,
                               args.atol, stretch)
    p_filter.initialize(x_0, args.n_initialize)
    p_filter.prune_hack(relaxed_x, 1.5 * args.initial_dx)

    # Run filter on y_q
    transition = 60
    p_filter.forward(y_q, 0, transition, gamma, clouds)
    epsilon_max = args.epsilon_min * args.epsilon_ratio
    p_filter.change_epsilon_stretch(args.epsilon_min, epsilon_max, stretch)
    for t_start in range(transition, len(y_q), 5):
        if cloud_marks[t_start]:
            p_filter.forward(y_q, t_start, t_start + 5, gamma, clouds)
        else:
            p_filter.forward(y_q, t_start, t_start + 5, gamma)

    # Write results

    with open(args.result, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
