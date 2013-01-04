'''lor_C.pyx: cython code for speed up.  Integrates the Lorenz system.

derived from hmm/C.pyx

From http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

Now pick a step-size h>0 and define

    \begin{align} y_{n+1} &= y_n + \tfrac{1}{6} \left(k_1 + 2k_2 + 2k_3 + k_4 \right)\\ t_{n+1} &= t_n + h \\ \end{align}

for n=0,1,2,3,... , using

    \begin{align} k_1 &= hf(t_n, y_n), \\ k_2 &= hf(t_n +
    \tfrac{1}{2}h , y_n + \tfrac{1}{2} k_1), \\ k_3 &= hf(t_n +
    \tfrac{1}{2}h , y_n + \tfrac{1}{2} k_2), \\ k_4 &= hf(t_n + h ,
    y_n + k_3). \end{align} [1]

'''
#To build: python3 setup.py build_ext --inplace


import numpy as np
# Imitate http://docs.cython.org/src/tutorial/numpy.html
cimport cython, numpy as np
DTYPE = np.float64
ITYPE = np.int32
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

cdef f_lor_c(double *x, double t, double s, double b, double r, double *y_dot):
    '''This function acts as if it were pure C.  It calculates the vector
    field (y_dot) at x,t for the Lorenz system with parmaters s, b, r.

    '''
    y_dot[0] = s * (x[1] - x[0])
    y_dot[1] = r * x[0] - x[0] * x[2] - x[1]
    y_dot[2] = x[0] * x[1] - b * x[2]
    return

cdef lor_step(
    double *x_i, # Initial position
    double h,    # Time step
    double *x_f, # Storage for final position
    double t,    # Time (not used)
    double s, double b, double r, # Parameters
):
    ''' Implements a fourth order Runge Kutta step using f_lor() for
        the vector field.
        '''
    cdef double k[5][3] # Storage for intermediate results.
    f_lor_c(x_i,t,s,b,r,k[0])
    for i in range(3):
        k[1][i] = h*k[0][i]
        x_f[i] = x_i[i] + k[1][i]/2

    f_lor_c(x_f,t,s,b,r,k[0])
    for i in range(3):
        k[2][i] = h*k[0][i]
        x_f[i] = x_i[i] + k[2][i]/2

    f_lor_c(x_f,t,s,b,r,k[0])
    for i in range(3):
        k[3][i] = h*k[0][i]
        x_f[i] = x_i[i] + k[3][i]

    f_lor_c(x_f,t,s,b,r,k[0])
    for i in range(3):
        k[4][i] = h*k[0][i]
        x_f[i] = x_i[i] + (k[1][i] + 2 * k[2][i] + 2 * k[3][i] + k[4][i])/6
    return

tmp = np.empty(3)
@cython.boundscheck(False)
def f_lor(np.ndarray[DTYPE_t, ndim=1] x, double t, double s, double b,
          double r):
    ''' Evaluates dy/dt at x for the Lorenz system.  Called from python.
    '''
    cdef np.ndarray[DTYPE_t, ndim=1] _tmp = tmp
    cdef double *y_dot = <double *>_tmp.data
    cdef double *X = <double *>x.data
    f_lor_c(X, t, s, b, r, y_dot)
    return tmp

def Lsteps(IC,      # IC[0:3] is the initial condition
           s, b, r, # These are the Lorenz parameters
           T_step,  # The time between returned samples
           N_steps  # N_steps The number of returned samples
           ):
    rv = np.empty((N_steps, 3))               # Storage for result
    cdef np.ndarray[DTYPE_t, ndim=2] _rv = rv
    cdef double *rv_p = <double *>_rv.data    # C like pointer to result
    rv[0,:] = IC

    scratch = np.empty(6)
    cdef np.ndarray[DTYPE_t, ndim=1] _scratch = scratch
    cdef double *scratch_p = <double *>_scratch.data # temporary data
    cdef double t = 0, *x_i, *x_f, *x_p
    cdef int i, j, N
    N = max(1,(T_step/0.005))
    cdef double h = T_step/N
    for i in range(1,N_steps):
        x_i = scratch_p
        x_f = scratch_p+3
        for j in range(3):
            x_i[j] = rv_p[(i-1)*3+j]
        for j in range(N-1):
            lor_step(x_i, h, x_f, t, s, b, r)
            x_p = x_f
            x_f = x_i
            x_i = x_p
        lor_step(x_i, h, <double *> (&rv_p[i*3]), t, s, b, r)
    return rv

#--------------------------------
# Local Variables:
# mode: python
# End:
