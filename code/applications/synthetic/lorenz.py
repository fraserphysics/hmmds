'''lorenz.py

This file may be imported into other scripts to provide a python
interface to gsl for integrating the lorenz system.  It may also be
called as "main" to make data files.  Here is the Lorenz system:

   \dot x = s(y-x)
   \dot y = rx -xz -y
   \dot z = xy -bz

Here are the fuctions that the original version of this module provided:
Lsteps, Ltan_steps, Ltan_one

>>> IC = np.array([0.1,0.3,2.0])
>>> x = Lsteps(IC,10.0,8.0/3,28.0,0.01,4)
>>> for y in x:
...     print('%5.2f %5.2f %5.2f'%tuple(y))
 0.10  0.30  2.00
 0.12  0.33  1.95
 0.14  0.36  1.90
 0.16  0.39  1.85
'''
import sys, numpy as np

def F(x,t,s,b,r):
    return np.array([
        s*(x[1]-x[0]),
        x[0]*(r - x[2])-x[1],
        x[0]*x[1]-b*x[2]
        ])
def Lsteps(IC,      # IC[0:3] is the initial condition
           s, b, r, # These are the Lorenz parameters
           T_step,  # The time between returned samples
           N_steps  # N_steps The number of returned samples
           ):
    from scipy.integrate import odeint
    t = np.arange(N_steps,dtype=float)*T_step
    return odeint(F, np.array(IC), t, args=(s,b,r))
def main(argv=None):
    import argparse

    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='Make files derived from Lorenz simulations')
    parser.add_argument('--L', type=int, default=100,
                       help='Number of samples')
    parser.add_argument('--s', type=float, default=10.0,
                       help='Lorenz s parameter')
    parser.add_argument('--r', type=float, default=28.0,
                       help='Lorenz r parameter')
    parser.add_argument('--b', type=float, default=8.0/3,
                       help='Lorenz b parameter')
    parser.add_argument('--dt', type=float, default=0.15,
                       help='Sample interval')
    parser.add_argument('--levels', type=int, default=4,
                        help='Number of quatization levels')
    parser.add_argument('--quantfile', type=argparse.FileType('w'),
                       help='Write quantized data to this file')
    parser.add_argument('--xyzfile', type=argparse.FileType('w'),
                       help='Write x,y,z data to this file')
    args = parser.parse_args(argv)
    args_d = args.__dict__
    if args.xyzfile==None and args.quantfile==None:
        import doctest
        doctest.testmod()
    else:
        print('Something to calculate')
    return 0

if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# eval: (python-mode)
# End:
