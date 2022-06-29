r"""lorenz.py

The functions in this module may be imported into other scripts to
provide a python interface to tools for integrating the lorenz system
(eg, scipy.integrate or gsl).  It may also be called as "main" to make
data files.  Here is the Lorenz system

.. math::
    \dot x = s(y-x)

    \dot y = rx -xz -y

    \dot z = xy -bz

"""
import sys
import argparse
import os.path

import numpy
import scipy.integrate  # type: ignore


def parse_args(argv):
    """Parse the command line.
    """
    parser = argparse.ArgumentParser(
        description='Make files derived from Lorenz simulations')
    parser.add_argument('--n_samples',
                        type=int,
                        default=100,
                        help='Number of samples')
    parser.add_argument('--IC',
                        type=float,
                        nargs=3,
                        default=[11.580, 13.548, 28.677],
                        help='Initial conditions')
    parser.add_argument('--s',
                        type=float,
                        default=10.0,
                        help='Lorenz s parameter')
    parser.add_argument('--r',
                        type=float,
                        default=28.0,
                        help='Lorenz r parameter')
    parser.add_argument('--b',
                        type=float,
                        default=8.0 / 3,
                        help='Lorenz b parameter')
    parser.add_argument('--dt',
                        type=float,
                        default=0.15,
                        help='Sample interval')
    parser.add_argument('--levels',
                        type=int,
                        default=4,
                        help='Number of quatization levels')
    parser.add_argument('--quantfile',
                        type=argparse.FileType('w'),
                        help='Write quantized data to this file')
    parser.add_argument('--xyzfile',
                        type=argparse.FileType('w'),
                        help='Write x,y,z data to this file')
    parser.add_argument('--TSintro', help='Directory to write data to')
    return parser.parse_args(argv)


def lorenz_dx_dt(x, _, s, b, r):  # pylint: disable=invalid-name
    """ Lorenz vector field

    Args:
        x (numpy.ndarray): State vector
        _ (float): Time.  Unused, but necessary argument for scipy.integrate.odeint
        s (float): Parameter of Lorenz model
        b (float): Parameter of Lorenz model
        r (float): Parameter of Lorenz model

    Return:
        (numpy.ndarray): dx/dt at x for Lorenz model.
    """
    return numpy.array(
        [s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]])


def lorenz_tangent(  # pylint: disable=invalid-name
        x_aug,  # 12 dimensional augmented vector
        t,
        s,
        b,
        r):
    """ Lorenz vector field augmented with tangent space

    Args:
        x_aug (numpy.ndarray): State vector and flattened 3x3 dx(0)/dx(t)
        t (float): Unused, but necessary argument for scipy.integrate.odeint
        s (float): Parameter of Lorenz model
        b (float): Parameter of Lorenz model
        r (float): Parameter of Lorenz model

    Return:
        (numpy.ndarray): dx/dt at x augmented with d/dt (dx(t)/dx(0))
            for Lorenz model.
    """
    return_value = numpy.empty(12)
    x = x_aug[:3]

    # First three components are the value F(x)
    return_value[:3] = lorenz_dx_dt(x, t, s, b, r)

    dF = numpy.array([  # The derivative of F wrt x
        [-s, s, 0], [r - x[2], -1, -x[0]], [x[1], x[0], -b]
    ])

    # Prepare tangent part for dot
    dx_dx0 = x_aug[3:].reshape((3, 3))

    # Assign the tangent part of the return value.
    return_value[3:] = numpy.dot(dF, dx_dx0).reshape(-1)

    return return_value


def main(argv=None):
    """Writes time series to files specified by options --xyzfile,
    --quantfile, and or --TSintro.

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    lorenz_args = (args.s, args.b, args.r)
    initial_conditions = numpy.array(args.IC)

    def t_array(n_times, delta_t):
        """Return an array of uniformly spaced sample times for odeint.

        Args:
            n_times: Number of samples
            delta_t: Interval between sequential samples
        """
        return numpy.arange(n_times, dtype=float) * delta_t

    xyz = scipy.integrate.odeint(lorenz_dx_dt, initial_conditions,
                                 t_array(args.n_samples, args.dt), lorenz_args)

    # Calculate quantization parameters.  Use ceil and floor so that
    # quantization will be the same for most long series.  The
    # quantization results will range from 1 to args.levels including
    # the end points.  I use 1 for the minimum so that plots look
    # nice.
    _max = xyz[:, 0].max()
    _min = xyz[:, 0].min()
    scale = numpy.ceil((_max - _min) / args.levels)
    offset = numpy.floor(_min / scale)

    def quant(x):
        return int(numpy.ceil(x / scale - offset))

    assert quant(_max) == args.levels
    assert quant(_min) == 1

    for vector in xyz:
        # pylint: disable = consider-using-f-string
        print('{0:6.3f} {1:6.3f} {2:6.3f}'.format(*vector), file=args.xyzfile)
        print(f'{quant(vector[0]):d}', file=args.quantfile)
    if args.TSintro is not None:
        xyz = scipy.integrate.odeint(lorenz_dx_dt, initial_conditions,
                                     t_array(args.n_samples, args.dt / 50),
                                     lorenz_args)
        # Write x[0] to TSintro_fine with time step .003
        with open(os.path.join(args.TSintro, 'fine'),
                  encoding='utf-8',
                  mode='w') as fine:
            for i in range(0, args.n_samples):
                fine.write(f'{i*args.dt/50:6.3f} {xyz[i,0]:6.3f}\n')

        # Write x[0] to TSintro_qt with time step .15
        with open(os.path.join(args.TSintro, 'coarse'),
                  encoding='utf',
                  mode='w') as coarse:
            for i in range(0, args.n_samples, 50):
                coarse.write(f'{i*args.dt/50:6.3f} {xyz[i,0]:6.3f}\n')

        # Write quantized x[0] to TSintro_qtx with time step .15
        quantized_data = numpy.ceil(xyz[:, 0] / 10 + 2)
        with open(os.path.join(args.TSintro, 'quantized'),
                  encoding='utf',
                  mode='w') as quantized:
            for i in range(0, args.n_samples, 50):
                quantized.write(f'{int(i/50):2d} {quantized_data[i]:6.3f}\n')
    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
