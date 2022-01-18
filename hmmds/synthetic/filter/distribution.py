r"""distibution.py

Characterize the distribution of states for the following state space
model:

.. math::
    x_{t+1} = A x_t + B V_n

    y_t = C x_t + D W_n

    A = [cos(\omega * dt) sin(\omega * dt)] * \exp(-a dt)
        [-sin(\omega * dt) cos(\omega * dt)]

    B = [b 0]
        [0 b]

    C = [c 0]

V and W are unit variance iid Gaussian noise with dimension 2 and 1
respectively and the parameters \omega, dt, a, b, c, and d are
arguments to the module.

After relaxing, the following equation gives the scale of x

   E(x^2) = \frac{b^2}{1 - e^{-2*a*dt}}

"""

import sys
import argparse

import numpy
import numpy.random

import hmm.state_space
import plotscripts.utilities


def parse_args(argv):
    """Parse the command line.
    """
    period = 1.0
    omega = (2 * numpy.pi) / period

    parser = argparse.ArgumentParser(
        description=
        'Characterize the distribution of states of a stochastic system.')
    parser.add_argument('--sample_rate',
                        type=float,
                        default=5.0,
                        help='number of samples per cycle')
    parser.add_argument('--n_samples',
                        type=int,
                        default=1000,
                        help='Number of samples')
    parser.add_argument('--initial_state',
                        type=float,
                        nargs=2,
                        default=[0, 0],
                        help='Initial state')
    parser.add_argument('--omega',
                        type=float,
                        default=omega,
                        help='system rotation rate')
    parser.add_argument('--a',
                        type=float,
                        default=0.001 * omega,
                        help='system dissipation rate')
    parser.add_argument('--b',
                        type=float,
                        default=.01,
                        help='System noise multiplier')
    parser.add_argument('--c', type=float, default=0.5, help='Observation map')
    parser.add_argument('--d',
                        type=float,
                        default=0.2,
                        help='Observation noise multiplier')
    parser.add_argument('--random_seed', type=int, default=3)
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('fig_path', type=str, help="path to figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Writes time series to files specified by options --xyzfile,
    --quantfile, and or --TSintro.

    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    rng = numpy.random.default_rng(args.random_seed)

    def make_system(args, dt):
        """Make a system instance

        Args:
            args: Command line arguments
            dt: Sample interval

        Returns:
            A system instance
        """
        # pylint: disable = invalid-name
        a = numpy.array([[
            numpy.cos(args.omega * dt),
            numpy.sin(args.omega * dt)
        ], [-numpy.sin(args.omega * dt),
            numpy.cos(args.omega * dt)]]) * numpy.exp(-args.a * dt)
        # The sqrt(dt) factor in b enables changing dt without
        # changing the variance of x.  The variance of x is
        # proportional to args.b**2
        b = numpy.eye(2) * args.b * numpy.sqrt(
            dt)  # State noise is b * Normal(0,I), covariance is b*b
        c = numpy.array([
            [args.c, 0.0],
        ])
        d = numpy.array(
            [args.d],
            dtype=numpy.float64)  # Observation noise is c * Normal(0,I)
        return hmm.state_space.LinearGaussian(a, b, c, d,
                                              rng), args.a * dt, b[0, 0]

    def simulate(n_samples: int):
        """Draw a scalar time series and estimate variance.

        Args:
            system: A linear system
            n_samples: Number of samples to return
        """

        dt = 2 * numpy.pi / (args.omega * args.sample_rate)
        system, a_dt, b = make_system(args, dt)

        mean = numpy.zeros(2)
        covariance = numpy.eye(2) * args.b**2 * dt

        x01, _ = system.simulate_n_steps(
            hmm.state_space.MultivariateNormal(mean, covariance, rng),
            args.n_samples)
        variance = (x01[:, 0]**2).sum() / args.n_samples
        return x01, variance, a_dt, b

    variance = []
    a_dts = []
    as_ = numpy.linspace(.002, .02, 11)
    for a in as_:
        args.a = a
        x01, variance_, a_dt, b_root_dt = simulate(args.n_samples)
        variance.append(variance_)
        a_dts.append(a_dt)

    fig, (axis_a, axis_b) = pyplot.subplots(nrows=2, figsize=(6, 8))

    #histogram, edges = numpy.histogram(x[:,0], bins=100)
    # axis_b.plot(edges[:-1], numpy.log(histogram))

    a_dts = numpy.array(a_dts)
    axis_a.plot(a_dts, variance)

    axis_a.plot(a_dts, b_root_dt**2 / (1 - numpy.exp(-2 * a_dts)))

    axis_a.set_xlabel('a\_dts')
    axis_a.set_ylabel('variance')

    axis_b.plot(x01[::100, 0])
    if args.show:
        pyplot.show()
    fig.savefig(args.fig_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
