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
    double *k_1, # Storage for intermediate result
    double *k_2, # Storage for intermediate result
    double *k_3, # Storage for intermediate result
    double *k_4, # Storage for intermediate result
    double *y_dot# Storage for intermediate result
):

    f_lor_c(x_i,t,s,b,r,y_dot)
    for i in range(3):
        k_1[i] = h*y_dot[i]
        x_f[i] = x_i[i] + k_1[i]/2

    f_lor_c(x_f,t,s,b,r,y_dot)
    for i in range(3):
        k_2[i] = h*y_dot[i]
        x_f[i] = x_i[i] + k_2[i]/2

    f_lor_c(x_f,t,s,b,r,y_dot)
    for i in range(3):
        k_3[i] = h*y_dot[i]
        x_f[i] = x_i[i] + k_3[i]

    f_lor_c(x_f,t,s,b,r,y_dot)
    for i in range(3):
        k_4[i] = h*y_dot[i]
        x_f[i] = x_i[i] + (k_1[i] + 2 * k_2[i] + 2 * k_3[i] + k_4[i])/6
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
    rv = np.empty((N_steps, 3))
    k_np = np.empty(21)
    cdef np.ndarray[DTYPE_t, ndim=1] k_ = k_np
    cdef double *k_pt = <double *> k_.data
    cdef double *k[7]
    cdef int i
    for i in range(7):
        k[i] = &k_pt[i*3]
    cdef double t = 0
    cdef double h = T_step
    cdef np.ndarray[DTYPE_t, ndim=2] _rv = rv
    cdef double *rv_p = <double *>_rv.data
    cdef double *x_i
    cdef double *x_f
    rv[0,:] = IC
    for i in range(1,N_steps):
        x_i = <double *> (&rv_p[(i-1)*3])
        x_f = k[5]
        lor_step(x_i, h/10, x_f, t, s, b, r, k[0], k[1], k[2], k[3], k[4])
        for j in range(4):
            x_i = k[5]
            x_f = k[6]
            lor_step(x_i, h/10, x_f, t, s, b, r, k[0], k[1], k[2], k[3], k[4])
            x_i = k[6]
            x_f = k[5]
            lor_step(x_i, h/10, x_f, t, s, b, r, k[0], k[1], k[2], k[3], k[4])
        x_i = k[5]
        x_f = <double *> (&rv_p[i*3])
        lor_step(x_i, h/10, x_f, t, s, b, r, k[0], k[1], k[2], k[3], k[4])
    return rv

#--------------------------------
# Local Variables:
# mode: python
# End:
