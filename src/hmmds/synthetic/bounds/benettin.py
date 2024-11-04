"""benettin_data.py makes data for fig:benettin illustrating Lyapunov
exponent convergence

Call with: python benettin_data.py result_file

plotscripts/bounds/benettin.py uses the r_run_time data produced here
to make fig:benettin.  Each of the two sub-plots in the figure use the
same r_run_time data.  The plotscript uses eq:LEaug from the book to
estimate the effect of noise on the estimates to make the lower plots.

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
                        help=r'Quantization resolution, \Delta in the book')
    parser.add_argument('--atol',
                        type=float,
                        default=1e-7,
                        help='Absolute error tolerance for integrator')
    parser.add_argument(
        '--perturbation',
        type=float,
        default=1.0,
        help=
        'Standard deviation of perturbation of initial condition for different runs'
    )
    parser.add_argument('--t_relax',
                        type=float,
                        default=10.0,
                        help='Time to move to attractor')
    parser.add_argument('--t_run',
                        type=float,
                        default=150.0,
                        help='Length of noisy time series')
    parser.add_argument('--n_runs', type=int, default=1000)
    parser.add_argument('--time_step', type=float, default=0.15)

    parser.add_argument(
        '--t_estimate',
        type=float,
        default=1500.0,
        help='Length of series for estimating lyapunov spectrum')
    parser.add_argument('result', type=str, help='write result to this path')
    return parser.parse_args(argv)


def relax(args, initial_state):
    """Integrate initial state forward to get on attractor
    """
    Q = numpy.eye(3)  # pylint: disable=invalid-name
    x = initial_state.copy()
    n_relax = int(args.t_relax / args.time_step)
    for _ in range(n_relax):
        x, _ = hmmds.synthetic.bounds.lorenz.integrate_tangent(args.time_step,
                                                               x,
                                                               Q,
                                                               atol=args.atol)
    return x, Q


def one_run(n_times, initial_distribution, state_noise,
            args: argparse.Namespace):
    """ Return a record of a Lyapunov exponent calculation.

    Args:
        n_times: Number of sample times to simulate
        initial_distribution: For drawing initial states
        state_noise: For drawing samples of state noise
        args: Holds parameters from the command line

    Return:
        r_t: Diagonal elements of R from QR decomposition at each time
    """
    r_t = numpy.empty((n_times, 3))
    x, Q = relax(args, initial_distribution.draw())  # pylint: disable=invalid-name
    # Get a random initial state on the attractor by drawing a
    # randomly perturbed initial state and relaxing back to the
    # attractor

    # Explanation of Bennetin algorithm:  Let
    # d_t = (d x[t]/d x[t-1])
    # q_t * r_t = d_t * q_{t-1}
    # q_{-1} = 1

    # Then q_1 * r_1 = d_1 * q_0, and q_0 * r_0 = d_0 * 1 and q_1 *
    # r_1 * r_0 = d_1 * d_0

    # Similarly q_n (r_n * r_{n-1} * ... * r_0) = (d x[n]/d x[0])

    for t in range(n_times):
        # Start with q_t for t-1.  Note that F, the integral of the
        # tangent, is linear, and so F(Id) * q_t = F(q_t)
        x, derivative = hmmds.synthetic.bounds.lorenz.integrate_tangent(
            args.time_step, x, Q, atol=args.atol)
        Q, R = numpy.linalg.qr(derivative)  # pylint: disable=invalid-name
        r_t[t] = numpy.abs(R.diagonal())
        assert r_t[t].min() > 0.0
        x += state_noise.draw()
    return r_t


def noiseless_lyapunov_spectrum(initial_state, args: argparse.Namespace):
    """ This is one_run without noise for estimating the lyapunov exponents

    Args:
        initial_state: For drawing initial states
        args: Holds parameters from the command line

    Return:
        lambda: Estimates of the 3 lyapunov exponents
    """
    sum_log_r = numpy.zeros(3)
    # pylint: disable=invalid-name
    x, Q = relax(args, initial_state)

    n_times = int(args.t_estimate / args.time_step)
    for _ in range(n_times):
        x, derivative = hmmds.synthetic.bounds.lorenz.integrate_tangent(
            args.time_step, x, Q, atol=args.atol)
        Q, R = numpy.linalg.qr(derivative)  # pylint: disable=invalid-name
        r = numpy.abs(R.diagonal())
        assert r.prod() > 0.0
        sum_log_r += numpy.log(r)
    spectrum = sum_log_r / args.t_estimate
    print(f'{spectrum=}')
    return spectrum


def lyapunov_spectrum_with_noise(args, r_run_time: numpy.ndarray) -> dict:
    """ Estimate characteristics of distribution of estimates

    Args:
        args: Command line arguments
        r_run_time: Diagonal elements of R from QR decompositions

    Return:
        statistics
    """
    (n_runs, _, three) = r_run_time.shape
    assert three == 3

    log_r = numpy.log(r_run_time).sum(axis=1) / args.t_run
    assert log_r.shape == (n_runs, 3)
    mean = numpy.mean(log_r, axis=0)
    assert mean.shape == (3,)
    std = numpy.std(log_r, axis=0, ddof=1)

    print(f'{mean=}\n {std=}')

    return {'mean': mean, 'std': std}


def main(argv=None):
    """Study Lyuponov exponent calculation.

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    # Relax to a point near the attractor
    relaxed_x = relax(args, numpy.ones(3))[0]

    # Set up generators for initial conditions and state noise
    initial_distribution = hmm.state_space.MultivariateNormal(
        relaxed_x,
        numpy.eye(3) * args.perturbation**2, rng)
    state_noise = hmm.state_space.MultivariateNormal(
        numpy.zeros(3),
        numpy.eye(3) * args.dev_state**2, rng)

    n_times = int(args.t_run / args.time_step)
    r_run_time = numpy.empty((args.n_runs, n_times, 3))
    for n_run in range(args.n_runs):
        r_run_time[n_run] = one_run(n_times, initial_distribution, state_noise,
                                    args)

    result = lyapunov_spectrum_with_noise(args, r_run_time)
    result['r_run_time'] = r_run_time
    result['args'] = args
    result['spectrum'] = noiseless_lyapunov_spectrum(relaxed_x, args)
    with open(args.result, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:
