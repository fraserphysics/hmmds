r"""distibution.py

Characterize the distribution of states for the state space model defined in linear_simulatinon.py

"""

import sys
import argparse

import numpy
import numpy.random
import scipy.stats

import hmm.state_space
import plotscripts.utilities

import linear_simulation


def parse_args(argv):
    """Parse the command line.
    """

    parser = argparse.ArgumentParser(
        description=
        'Characterize the distribution of states of a stochastic system.')
    linear_simulation.system_args(parser)

    parser.add_argument('--n_samples',
                        type=int,
                        default=1000,
                        help='Number of samples')
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
        return x01, variance, dt

    x01, variance, dt = simulate(args.n_samples)
    x0 = x01[:, 0]

    fig, (axis_a, axis_b) = pyplot.subplots(nrows=2, figsize=(6, 8))

    axis_a.plot(numpy.array(range(len(x0))) * dt,
                x0,
                marker='.',
                linestyle='None')
    #axis_a.plot(x0)
    axis_a.set_xlabel('t')
    axis_a.set_ylabel('$x_0$')

    quantiles, sorted = scipy.stats.probplot(x0,
                                             dist='norm',
                                             sparams=(0.0, variance**.5),
                                             fit=False)
    #res = scipy.stats.probplot(x0, fit=True, dist='norm', sparams=(0.0, .5*variance**.5), plot=axis_b)
    axis_b.plot(quantiles, sorted, marker='.', linestyle='None')
    ends = [quantiles[0], quantiles[-1]]
    axis_b.plot(ends, ends)
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
