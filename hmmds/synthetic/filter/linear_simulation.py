r"""linear_simulation.py

Generate a sequence of observations from the following state space
model
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

"""

import sys
import argparse

import numpy
import numpy.random

import hmm.state_space

def parse_args(argv):
    """Parse the command line.
    """
    period = 1.0
    omega = (2*numpy.pi)/period

    parser = argparse.ArgumentParser(
        description='Generate a sequence of observations from a state space model.')
    parser.add_argument('--n_samples',
                        type=int,
                        default=1000,
                        help='Number of samples')
    parser.add_argument('--mean',
                        type=float,
                        nargs=2,
                        default=[0,0],
                        help='Initial mean')
    parser.add_argument('--covariance',
                        type=float,
                        nargs=3,
                        default=[.25, 0, .25],
                        help='Initial covariance components (1,1), (1,2), and (2,2)')
    parser.add_argument('--omega',
                        type=float,
                        default=omega,
                        help='system rotation rate')
    parser.add_argument('--a',
                        type=float,
                        default=0.001*omega,
                        help='system dissipation rate')
    parser.add_argument('--dt',
                        type=float,
                        default=period/10.0,
                        help='Sample frequency')
    parser.add_argument('--b',
                        type=float,
                        default=.1,
                        help='System noise multiplier')
    parser.add_argument('--c',
                        type=float,
                        default=0.5,
                        help='Observation map')
    parser.add_argument('--d',
                        type=float,
                        default=0.2,
                        help='Observation noise multiplier')
    parser.add_argument('--random_seed', type=int, default=3)
    parser.add_argument('xfile',
                        type=argparse.FileType('w', encoding='utf-8'),
                        help='Write x data to this file')
    parser.add_argument('yfile',
                        type=argparse.FileType('w', encoding='utf-8'),
                        help='Write y data to this file')
    parser.add_argument('filtered',
                        type=argparse.FileType('w', encoding='utf-8'),
                        help='Write filtered data to this file')
    parser.add_argument('covariances',
                        type=argparse.FileType('w', encoding='utf-8'),
                        help='Write filtered data to this file')
    return parser.parse_args(argv)

def main(argv=None):
    """Writes time series to files specified by options --xyzfile,
    --quantfile, and or --TSintro.

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    rng = numpy.random.default_rng(args.random_seed)

    # pylint: disable = invalid-name
    a = numpy.array([
        [numpy.cos(args.omega*args.dt), numpy.sin(args.omega*args.dt)],
        [-numpy.sin(args.omega*args.dt), numpy.cos(args.omega*args.dt)]
        ]) * numpy.exp(-args.a * args.dt)
    b = numpy.eye(2) * args.b  # State noise is b * Normal(0,I)
    c = numpy.array([[args.c, 0.0],])
    d = numpy.array([args.d],dtype=numpy.float64) # Observation noise is c * Normal(0,I)
    mean = numpy.array(args.mean)
    covariance = numpy.array([
        [args.covariance[0], args.covariance[1]],
        [args.covariance[1], args.covariance[0]]
        ])

    initial_dist = hmm.state_space.MultivariateNormal(mean, covariance, rng)
    system = hmm.state_space.LinearGaussian(a, b, c, d, rng)

    x, y = system.simulate_n_steps(initial_dist, args.n_samples)
    means, covariances = system.filter(initial_dist, y)  # Run Kalman filter on simulated observations

    # Write the results
    for t in range(0, args.n_samples):
        args.xfile.write(f'{x[t,0]:6.3f} {x[t,1]:6.3f}\n')

    for t in range(0, args.n_samples):
        args.filtered.write(f'{means[t,0]:6.3f} {means[t,1]:6.3f}\n')

    for t in range(0, args.n_samples):
        args.covariances.write(f"""{t:3d} {covariances[t,0,0]:6.3f} {covariances[t,0,1]:6.3f}
    {covariances[t,1,0]:6.3f} {covariances[t,1,1]:6.3f}\n""")

    for t in range(0, args.n_samples):
        args.yfile.write(f'{y[t,0]:6.3f}\n')
    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
