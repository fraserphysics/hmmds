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
import copy

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
    """A point in the Lorenz system with a box (parallelogram) and a weight.

    Args:
        x: 3-vector position
        box: 3x3 derivative matrix
        weight: Scalar
        parent: For visualizing ancestry
        neighbor: Vector to neighbor

    One corner of the box is at x.  Other corners are given by x +
    box.

    """

    # pylint: disable=invalid-name
    def __init__(self: Particle, x, box, weight, parent, neighbor):
        self.x = x
        self.box = box
        self.weight = weight
        self.parent = parent
        self.neighbor = neighbor

    def step(self: Particle, time, atol):
        """Map the box forward by the interval "time"

        Args:
            time: The amount of time
            atol: Integration absolute error tolerance
        """
        pre_neighbor = numpy.linalg.lstsq(self.box, self.neighbor,
                                          rcond=1e-6)[0]
        self.x, self.box = hmmds.synthetic.bounds.lorenz.integrate_tangent(
            time, self.x, self.box, atol=atol)
        self.neighbor = numpy.dot(self.box, pre_neighbor)
        assert self.box.shape == (3, 3)
        assert self.neighbor.shape == (3,)

    def divide(self: Particle, n_divide, U, S, VT):
        """Divide self into n_divide new particles along S_0 direction

        Args:
            n_divide: number of new particles
            U, S, VT: Singular value decomposition of self.box
        """
        assert n_divide > 0
        S[0] /= n_divide
        new_box = U * S
        x_step = U[:, 0] * S[0] * VT[0, :]
        new_weight = self.weight / n_divide

        result = [
            Particle(self.x + i * x_step, new_box, new_weight, self.parent,
                     x_step) for i in range(n_divide)
        ]
        return result


class Filter:
    """Variant of particle filter using Lorenz equations for discrete
    observations

    Args:
        epsilon_min: Edge length of small box
        epsilon_max: Failure of linear approximation gives maximum length
        n_min: Minimum number of particles
        bins: Quatization boundaries for observations
        time_step: Integrate Lorenz this interval between samples
        atol: Absolute error tolerance for integrator
    

    """

    def __init__(self: Filter, epsilon_min, epsilon_max, bins, time_step, atol):
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.bins = bins
        self.time_step = time_step
        self.atol = atol
        self.particles = []

    def change_epsilon(self: Filter, new_min: float, new_max: float):
        """Change particle box sizes.
        
        Args:
            new_min:
            new_max

        """
        self.epsilon_min = new_min
        self.epsilon_max = new_max

    def initialize(self: Filter,
                   initial_x: numpy.ndarray,
                   n_times: int,
                   delta=None):
        """Populate self.particles by integrating Lorenz

        Args:
            initial_x: Integrate this 3-vector n_times for initial distribution
            n_times: Number of time steps
            delta: Length of edges of initial particles

        
        """
        if delta is None:
            delta = self.epsilon_max
        x_t = hmmds.synthetic.bounds.lorenz.n_steps(initial_x, n_times,
                                                    self.time_step, self.atol)
        keys, counts = numpy.unique(numpy.around(x_t / delta).astype(int),
                                    return_counts=True,
                                    axis=0)
        neighbor = numpy.array([delta, 0, 0])
        self.particles = [
            Particle(
                key * delta,  # Position
                numpy.eye(3) * delta,  # Initial box
                count,  # Number of times occured
                parent,  # ID for color plot
                neighbor  # vector to adjacent particle
            ) for parent, (key, count) in enumerate(zip(keys, counts))
        ]
        self.normalize()

    def forecast_x(self: Filter, time: float):
        """Map each particle forward by time.  If the largest singular
        value S[0] > epsilon_max, subdivide the particle.

        Args:
            time: Map via integrating Lorenz for this time step.

        """
        new_particles = []
        for particle in self.particles:
            particle.step(time, self.atol)
            U, S, VT = numpy.linalg.svd(particle.box)
            if S[0] < self.epsilon_max:
                new_particles.append(particle)
            else:
                new_particles.extend(
                    particle.divide(int(S[0] / self.epsilon_min), U, S, VT))
        self.particles = new_particles

    def update(self: Filter, y: int):
        """Delete particles that don't match y.

        Args:
            y: A scalar integer observation
        """
        new_particles = []
        for particle in self.particles:
            if numpy.digitize(particle.x[0], self.bins) == y:
                new_particles.append(particle)
        if len(self.particles) > 0 and len(new_particles) == 0:
            # Print error diagnostics
            for parent, count in zip(*numpy.unique(numpy.array(
                [particle.parent for particle in self.particles]),
                                                   return_counts=True)):
                print(f'{parent=} {count=}')
            print(f'In update {len(self.particles)=} {len(new_particles)=}')
        self.particles = new_particles

    def normalize(self: Filter):
        """Scale weights so that total is 1
        """
        total_weight = 0.0
        for particle in self.particles:
            total_weight += particle.weight
        for particle in self.particles:
            particle.weight /= total_weight

    def p_y(self: Filter):
        """Calculate probability mass function for possible y values.

        """
        result = numpy.zeros(len(self.bins) + 1)
        for particle in self.particles:
            y = numpy.digitize(particle.x, self.bins)
            result[y] += particle.weight
        return result

    def forward(self: Filter,
                y_ts: numpy.ndarray,
                t_start,
                t_stop,
                gamma,
                clouds=None):
        """Estimate and return gamma[t] = p(y[t] | y[0:t]) for t from t_start to t_stop.

        Args:
            y_ts: A time series of observations
            t_start:
            t_stop:
            gamma:
            clouds: Optional dict for saving particles

        """
        for t in range(t_start, t_stop):
            y = y_ts[t]
            print(f'{t=} {len(self.particles)=}')
            assert len(self.particles) < 1e7

            self.normalize()
            gamma[t] = self.p_y()[y]
            if clouds is not None:
                clouds[(t, 'forecast')] = copy.deepcopy(self.particles)
            self.update(y)
            if clouds is not None:
                clouds[(t, 'update')] = copy.deepcopy(self.particles)
            self.forecast_x(self.time_step)


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
