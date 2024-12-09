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
                        default=100.0,
                        help='Time to move to attractor')
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
    x_0 = benettin.relax(args, numpy.ones(3))[0]

    # Generate quantized data y_q and x_0 for filter.  Use
    # lorenz.integrate_tangent instead of lorenz.n_steps because
    # results from n_steps and integrate_tangent diverge from each
    # other too fast for filtering with integrate_tangent on data from
    # n_steps to work.
    x_list = [x_0]
    tangent = numpy.eye(3) * args.epsilon_min
    for n in range(args.n_y - 1):
        x_0, _ = lorenz.integrate_tangent(args.time_step,
                                          x_0,
                                          tangent,
                                          atol=args.atol)
        x_list.append(x_0)
    x_all = numpy.asarray(x_list)
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
    cloud_marks = make_marks(  #
        len(y_q),  #
        (  #
            (0, len(y_q)),  #
            # ((int(len(y_q) * .95)), len(y_q)),
        ))

    # Initialize filter
    stretch = 1.25
    epsilon_max = args.epsilon_min * args.epsilon_ratio
    p_filter = benettin.Filter(args.epsilon_min, epsilon_max, bins,
                               args.time_step, args.atol, stretch)
    p_filter.initialize(x_all[0], epsilon_max)
    scale = 1.0

    # Run filter on y_q
    for t_start in range(0, len(y_q), 5):
        if cloud_marks[t_start]:
            scale = p_filter.forward(y_q, t_start, t_start + 5, gamma, scale,
                                     clouds)
        else:
            scale = p_filter.forward(y_q, t_start, t_start + 5, gamma, scale)
        if len(p_filter.particles) == 0:
            break

    # Write results

    with open(args.result, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
