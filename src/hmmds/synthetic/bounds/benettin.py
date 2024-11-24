"""benettin_data.py makes data for fig:benettin illustrating Lyapunov
exponent convergence

Call with: python benettin_data.py result_file

plotscripts/bounds/benettin.py uses the r_run_time data produced here
to make fig:benettin.  Each of the two sub-plots in the figure use the
same r_run_time data.  The plotscript uses eq:LEaug from the book to
estimate the effect of noise on the estimates to make the lower plots.

"""

from __future__ import annotations  # Enables, eg, (self: Particle
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


class Particle:
    """A point in the Lorenz system with a box and a weight

    Args:
        x: 3-vector position
        Q: 3x3 matrix of QR decomposition defining a box
        R: 3-vector of QR decomposition defining a box
        weight: Scalar

    """

    # pylint: disable=invalid-name
    def __init__(self: Particle, x, Q, R, weight):
        self.x = x
        self.Q = Q
        self.R = R
        self.weight = weight

    def step(self: Particle, time, atol):
        self.x, derivative = hmmds.synthetic.bounds.lorenz.integrate_tangent(
            time, self.x, self.Q, atol=atol)
        self.Q, R = numpy.linalg.qr(derivative)  # pylint: disable=invalid-name
        self.R = numpy.matmul(R, self.R)
        assert self.R.shape == (3, 3)

    def divide(self: Particle, axis, n_divide):
        """Divide self into n new particles along axis and shrink by ratio

        Args:
            axis: Divide along this edge
            n_divide: number of new particles
        """
        column_axis = self.R[:, axis]
        step = column_axis / n_divide
        base = self.x - column_axis / 2 + step / 2

        new_R = self.R.copy()
        new_R[:, axis] = column_axis / n_divide
        new_weight = self.weight / n_divide
        return [
            Particle(base + i * step, self.Q, new_R, new_weight)
            for i in range(n_divide)
        ]


class Filter:
    """Variant of particle filter using Lorenz equations for discrete
    observations

    Args:
        initial_x: 3-vector
        epsilon_min: Edge length of small box
        epsilon_max: Failure of linear approximation gives maximum length
        n_min: Minimum number of particles
        n_nominal: Desire this number of particles
        bins: Quatization boundaries for observations
        time_step: Integrate Lorenz this interval between samples
        atol: Absolute error tolerance for integrator
    

    """

    def __init__(self: Filter, epsilon_min, epsilon_max, n_min, n_nominal, bins, time_step, atol):
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.n_min = n_min
        self.n_nominal = n_nominal
        self.bins = bins
        self.time_step = time_step
        self.atol = atol
        self.particles = []

    def initialize(self: Filter,
                   initial_x: numpy.ndarray,
                   n_times: int,
                   delta=None):
        """Populate self.particles by integrating Lorenz

        Args:
            initial_x: Initial 3-vector
            times: Sequence of float times

        Return n_particles
        
        """
        if delta is None:
            delta = self.epsilon_max
        x_t = hmmds.synthetic.bounds.lorenz.n_steps(initial_x, n_times,
                                                    self.time_step, self.atol)
        keys, counts = numpy.unique((x_t / delta).astype(int),
                                    return_counts=True,
                                    axis=0)
        self.particles = [
            Particle(key * delta, numpy.eye(3),
                     numpy.eye(3) * self.epsilon_min, count)
            for key, count in zip(keys, counts)
        ]
        self.normalize()

    def forecast_x(self: Filter, time: float):
        n_particles = len(self.particles)
        amplifier = 1.0
        if n_particles < self.n_min:
            amplifier = self.n_nominal/n_particles
        new_particles = []
        for particle in self.particles:
            particle.step(time, self.atol)
            # Find longest box edge
            diagonal = numpy.abs(particle.R.diagonal())
            axis = numpy.argmax(diagonal)
            if diagonal[axis] < self.epsilon_max and amplifier == 1.0:
                new_particles.append(particle)
                continue
            n_divide = int(amplifier * diagonal[axis]/self.epsilon_min)
            new_particles.extend(particle.divide(axis, n_divide))
        self.particles = new_particles

    def update(self: Filter, y: int):
        new_particles = []
        for particle in self.particles:
            if numpy.digitize(particle.x[0], self.bins) == y:
                new_particles.append(particle)
        self.particles = new_particles

    def normalize(self: Filter):
        total_weight = 0.0
        for particle in self.particles:
            total_weight += particle.weight
        for particle in self.particles:
            particle.weight /= total_weight

    def p_y(self: Filter):
        result = numpy.zeros(len(self.bins) + 1)
        for particle in self.particles:
            y = numpy.digitize(particle.x, self.bins)
            result[y] += particle.weight
        return result

    def forward(self: Filter, ys: numpy.ndarray, time_step: float):
        """"""
        clouds = []
        gamma = numpy.empty(len(ys))
        for i, y in enumerate(ys):
            print(f'{i=} {len(self.particles)=}')
            assert len(self.particles) < 1e6
                
            self.normalize()
            gamma[i] = self.p_y()[y]
            clouds.append(
                numpy.array([particle.x for particle in self.particles]))
            self.update(y)
            clouds.append(
                numpy.array([particle.x for particle in self.particles]))
            self.forecast_x(time_step)
        return gamma, clouds


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


def sde_spectrum(args, r_run_time: numpy.ndarray) -> dict:
    """ Estimate distribution characteristics from running sde

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

    return mean, std


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

    sde_mean, sde_std = sde_spectrum(args, r_run_time)
    augment = args.dev_state / args.grid_size
    augmented_mean, augmented_std = sde_spectrum(args, r_run_time + augment)
    result = {
        'sde_mean': sde_mean,
        'sde_std': sde_std,
        'augmented_mean': augmented_mean,
        'augmented_std': augmented_std
    }
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
