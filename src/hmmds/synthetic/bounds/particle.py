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
                        default=2e-2,
                        help='Minimumlength of box edges')
    parser.add_argument('--epsilon_max',
                        type=float,
                        default=1e-1,
                        help='Maximumlength of box edges')
    parser.add_argument('--initial_dx',
                        type=float,
                        default=1.0,
                        help='Cell size for initialization')
    parser.add_argument('--n_min',
                        type=int,
                        default=50,
                        help='Minimum number of particles')
    parser.add_argument('--n_nominal',
                        type=int,
                        default=500,
                        help='Nominal number of particles')
    parser.add_argument('--n_y',
                        type=int,
                        default=30,
                        help='Number of test observations')
    parser.add_argument('--n_quantized',
                        type=int,
                        default=4,
                        help='Cardinality of test observations')
    parser.add_argument('--time_step', type=float, default=0.15)
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


def main(argv=None):
    """Run particle filter on Lorenz data

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    epsilon_ratio = args.epsilon_max / args.epsilon_min

    # Relax to a point near the attractor
    relaxed_x = benettin.relax(args, numpy.ones(3))[0]

    # Generate quantized data y_q and x_0 for filter
    x_all = lorenz.n_steps(relaxed_x, args.n_y, args.time_step, atol=args.atol)
    assert x_all.shape == (args.n_y, 3)
    bins = numpy.linspace(-20, 20, args.n_quantized + 1)[1:-1]
    y_q = numpy.digitize(x_all[:, 0], bins)
    x_0 = x_all[-1]

    # Initialize filter
    epsilon_max = args.initial_dx
    epsilon_min = epsilon_max / epsilon_ratio
    p_filter = benettin.Filter(epsilon_min, epsilon_max, args.n_min,
                               args.n_nominal, bins, args.time_step, args.atol)
    p_filter.initialize(x_0, args.n_initialize)
    print(f'{len(p_filter.particles)=}')

    result = {
        'x_all':
            x_all,
        'y_q':
            y_q,
        'initial_positions':
            numpy.array([particle.x for particle in p_filter.particles])
    }

    # Run filter on y_q
    clouds = {}
    gamma = numpy.zeros(len(y_q))
    result['gamma'] = gamma
    p_filter.forward(y_q, 0, 5, gamma)
    p_filter.forward(y_q, 5, 10, gamma)
    p_filter.change_epsilon(args.epsilon_min, args.epsilon_max)
    p_filter.forward(y_q, 10, 45, gamma,clouds)
    p_filter.forward(y_q, 45, len(y_q), gamma)

    result['clouds'] = clouds

    # Write results

    with open(args.result, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
