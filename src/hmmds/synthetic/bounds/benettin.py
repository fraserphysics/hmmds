""" benettin_data.py makes data illustrating Lyapunov exponent
convergence

Call with: python benettin_data.py result_file

Derived from hmmds3 LaypPlot.py

"""

import sys
import argparse
import pickle

import numpy
import numpy.linalg
import numpy.random

import hmm.state_space
import hmmds.synthetic.bounds.lorenz


def parse_args(argv):
    """Parse a command line.
    """
    parser = argparse.ArgumentParser(
        description='Make data to illustrate Lyapunov exponenet calculation.')
    parser.add_argument('--random_seed',
                        type=int,
                        default=7,
                        help='For random number generator')
    parser.add_argument('--dev_state',
                        type=float,
                        default=1e-5,
                        help='Standard deviation of state noise')
    parser.add_argument('--grid_size',
                        type=float,
                        default=1e-3,
                        help='Quantization resolution')
    parser.add_argument('--perturbation',
                        type=float,
                        default=1.0,
                        help='Standard deviation of perturbation')
    parser.add_argument('--n_relax',
                        type=int,
                        default=50,
                        help='Number of sample times to move to attractor')
    parser.add_argument('--n_times',
                        type=int,
                        default=1000,
                        help='Length of time series')
    parser.add_argument('--n_runs', type=int, default=1000)
    parser.add_argument('--time_step', type=float, default=0.15)
    parser.add_argument('result', type=str, help='write result to this path')
    return parser.parse_args(argv)


def one_run(initial_distribution, state_noise, args: argparse.Namespace):
    """ Return a record of a Lyapunov exponent calculation.

    Args:
        initial_distribution: For drawing initial states
        state_noise: For drawing samples of state noise
        args: Holds parameters from the command line

    Return:
        r_t: Diagonal elements of R from QR decomposition at each time
    """
    r_t = numpy.empty((args.n_times, 3))
    Q = numpy.eye(3)  # pylint: disable=invalid-name
    # Get a random initial state on the attractor by drawing a
    # randomly perturbed initial state and relaxing back to the
    # attractor
    x = initial_distribution.draw()
    for _ in range(args.n_relax):
        x, _ = hmmds.synthetic.bounds.lorenz.integrate_tangent(
            args.time_step, x, Q)

    # Explanation of Bennetin algorithm:  Let
    # d_t = (d x[t]/d x[t-1])
    # q_t * r_t = d_t * q_{t-1}
    # q_{-1} = 1

    # Then q_1 * r_1 = d_1 * q_0, and q_0 * r_0 = d_0 * 1 and q_1 *
    # r_1 * r_0 = d_1 * d_0

    # Similarly q_n (r_n * r_{n-1} * ... * r_0) = (d x[n]/d x[0])
    for t in range(args.n_times):
        # Start with q_t for t-1.  Note that F, the integral of the
        # tangent, is linear, and so F(Id) * q_t = F(q_t)
        x, derivative = hmmds.synthetic.bounds.lorenz.integrate_tangent(
            args.time_step, x, Q)
        Q, R = numpy.linalg.qr(derivative)  # pylint: disable=invalid-name
        r_t[t] = numpy.abs(R.diagonal())
        assert r_t[t].prod() > 0.0
        x += state_noise.draw()
    return r_t


def main(argv=None):
    """Study Lyuponov exponent calculation.

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    # Relax to a point near the attractor
    x = numpy.ones(3)
    for _ in range(args.n_relax):
        x, _ = hmmds.synthetic.bounds.lorenz.integrate_tangent(
            args.time_step, x, numpy.eye(3))

    # Set up generators for initial conditions and state noise
    initial_distribution = hmm.state_space.MultivariateNormal(
        x,
        numpy.eye(3) * args.perturbation**2, rng)
    state_noise = hmm.state_space.MultivariateNormal(
        numpy.zeros(3),
        numpy.eye(3) * args.dev_state**2, rng)

    r_run_time = numpy.empty((args.n_runs, args.n_times, 3))
    for n_run in range(args.n_runs):
        r_run_time[n_run] = one_run(initial_distribution, state_noise, args)

    with open(args.result, 'wb') as _file:
        pickle.dump({'r_run_time': r_run_time, 'args': args}, _file)


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
