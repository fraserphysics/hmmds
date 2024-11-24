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
    parser.add_argument('--epsilon',
                        type=float,
                        default=1e-4,
                        help='Nominal length of box edges')
    parser.add_argument('--n_divide',
                        type=int,
                        default=10,
                        help='Number of child particles after division')
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
    parser.add_argument('--initial_dx',
                        type=float,
                        default=0.2,
                        help='Cell size for initialization')
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

    # Relax to a point near the attractor
    relaxed_x = benettin.relax(args, numpy.ones(3))[0]

    # Generate quantized data y_q and x_0 for filter
    x_all = lorenz.n_steps(relaxed_x, args.n_y, args.time_step, atol=args.atol)
    assert x_all.shape == (args.n_y, 3)
    bins = numpy.linspace(-20, 20, args.n_quantized + 1)[1:-1]
    y_q = numpy.digitize(x_all[:, 0], bins)
    x_0 = x_all[-1]

    # Initialize filter
    p_filter = benettin.Filter(args.epsilon, args.n_divide, bins,
                               args.time_step, args.atol)
    p_filter.initialize(x_0, args.n_initialize, args.initial_dx)
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
    result['gamma'], result['clouds'] = p_filter.forward(y_q, args.time_step)

    # Write results

    with open(args.result, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
