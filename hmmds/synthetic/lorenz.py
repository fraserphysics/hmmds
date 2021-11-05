"""lorenz.py

This functions in this module may be imported into other scripts to
provide a python interface to tools for integrating the lorenz system
(eg, scipy.integrate or gsl).  It may also be called as "main" to make
data files.  Here is the Lorenz system

.. math::
    \dot x = s(y-x)

    \dot y = rx -xz -y

    \dot z = xy -bz

"""
Copyright = """
Copyright 2021 Andrew M. Fraser

Dshmm is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

Dshmm is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

See the file gpl.txt in the root directory of the dshmm distribution
or see <http://www.gnu.org/licenses/>.
"""
import sys
import argparse
import os.path

import numpy
import scipy.integrate  # type: ignore


def lorenz_dx_dt(x, _, s, b, r):
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


def lorenz_tangent(
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
        (numpy.ndarray): dx/dt at x augmented with d/dt (dx(0)/dx(t))
            for Lorenz model.
    """
    rv = numpy.empty(12)
    x = x_aug[:3]

    # First three components are the value F(x)
    rv[:3] = lorenz_dx_dt(x, t, s, b, r)

    dF = numpy.array([  # The derivative of F wrt x
        [-s, s, 0], [r - x[2], -1, -x[0]], [x[1], x[0], -b]
    ])

    dx_dx0 = x_aug[3:].reshape((3, 3))
    rv[3:] = numpy.dot(dF, dx_dx0).reshape(-1)
    return rv


def main(argv=None):
    """Writes time series to files specified by options --xyzfile,
    --quantfile, and or --TSintro.

    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

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
    args = parser.parse_args(argv)

    lorenz_args = (args.s, args.b, args.r)
    initial_conditions = numpy.array(args.IC)

    def t_array(n, dt):
        return numpy.arange(n, dtype=float) * dt

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

    for v in xyz:
        print('{0:6.3f} {1:6.3f} {2:6.3f}'.format(*v), file=args.xyzfile)
        print('{0:d}'.format(quant(v[0])), file=args.quantfile)
    if args.TSintro is not None:
        xyz = scipy.integrate.odeint(lorenz_dx_dt, initial_conditions,
                                     t_array(args.n_samples, args.dt / 50),
                                     lorenz_args)
        # Write x[0] to TSintro_fine with time step .003
        f = open(os.path.join(args.TSintro, 'fine'), 'w')
        for i in range(0, args.n_samples):
            print('{0:6.3f} {1:6.3f}'.format(args.dt / 50 * i, xyz[i, 0]),
                  file=f)
        # Write x[0] to TSintro_qt with time step .15
        f = open(os.path.join(args.TSintro, 'coarse'), 'w')
        for i in range(0, args.n_samples, 50):
            print('{0:6.3f} {1:6.3f}'.format(args.dt / 50 * i, xyz[i, 0]),
                  file=f)
        # Write quantized x[0] to TSintro_qtx with time step .15
        q = numpy.ceil(xyz[:, 0] / 10 + 2)
        f = open(os.path.join(args.TSintro, 'quantized'), 'w')
        for i in range(0, args.n_samples, 50):
            print('{0:2d} {1:6.3f}'.format(int(i / 50), q[i]), file=f)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
