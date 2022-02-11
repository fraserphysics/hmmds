#cython: language_level=3
"""lorenz_sde.pyx: cython code for speed up.

See http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

"""

import sys

import numpy

cimport cython  # For the boundscheck decorator
cimport numpy
DTYPE = numpy.float64
ctypedef numpy.float64_t DTYPE_t

def lorenz_integrate(
        numpy.ndarray[DTYPE_t, ndim=1] x_initial,
        float t_initial,
        float t_final,
        float h_max = 0.0025,
        float h_min =-0.001):
    """Integrate lorenz system from (x_initial, t_initial) to t_final.

    Args:
        x_initial: Initial state
        t_initial: Initial time
        t_final: Final time (Note t_final < t_initial is OK)
        h_max: Largest size of Runge-Kutta step
        h_min: Smallest size of Runge-Kutta step (Requre h_min < 0)

    Returns:
        x_final: The state at time t_final

    Experiments suggest that integrating forward for 1.0 units of time
    with h_max > .0025 gets far enough off of the attractor that
    integrating backwards for 1.0 units of time is pretty far off.
    Going backwards 0 > h_min > -0.001 seems OK for 1.0 units of time.

    """
    
    s, r, b = (10.0, 28.0, 8.0 / 3)
    assert h_min < 0 < h_max
    x_final = numpy.empty(3)

    # Get "memoryviews".  These enable calling functions with c style arguments
    cdef double[::1] initial_view = x_initial
    cdef double[::1] final_view = x_final

    # Calculate number of Runge-Kutta steps
    delta_t = t_final - t_initial
    if h_min < delta_t < h_max:
        lorenz_step(&initial_view[0], &final_view[0], delta_t, s, b, r)
        return x_final
    if delta_t > 0:
        n_steps = int(delta_t/h_max) + 1
    else:
        n_steps = int(delta_t/h_min) + 1
    assert n_steps > 1
    
    # Call multi-step integrate function
    lorenz_n_steps(&initial_view[0], &final_view[0], delta_t/n_steps, s, b, r, n_steps)
    return x_final

def tangent_integrate(
        numpy.ndarray[DTYPE_t, ndim=1] x_initial,
        float t_initial,
        float t_final,
        float h_max = 0.0025,
        float h_min =-0.001):
    """Integrate tangent system from (x_initial, t_initial) to t_final.

    Args:
        x_initial: Initial vector of state and tangent
        t_initial: Initial time
        t_final: Final time (Note t_final < t_initial is OK)
        h_max: Largest size of Runge-Kutta step
        h_min: Smallest size of Runge-Kutta step (Requre h_min < 0)

    Returns:
        x_final: The vector at time t_final

    """
    
    s, r, b = (10.0, 28.0, 8.0 / 3)
    assert h_min < 0 < h_max
    x_final = numpy.empty(12)

    # Get "memoryviews".  These enable calling functions with c style arguments
    cdef double[::1] initial_view = x_initial
    cdef double[::1] final_view = x_final

    # Calculate number of Runge-Kutta steps
    delta_t = t_final - t_initial
    if h_min < delta_t < h_max:
        tangent_step(&initial_view[0], &final_view[0], delta_t, s, b, r)
        return x_final
    if delta_t > 0:
        n_steps = int(delta_t/h_max) + 1
    else:
        n_steps = int(delta_t/h_min) + 1
    assert n_steps > 1
    
    # Call multi-step integrate function
    tangent_n_steps(&initial_view[0], &final_view[0], delta_t/n_steps, s, b, r, n_steps)
    return x_final

@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector_field(double *x, double s, double b, double r,
             double *y_dot):
    """This function acts as if it were pure C.  It calculates the vector
    field (y_dot) at x,t for the Lorenz system with parmaters s, b, r.

    """
    y_dot[0] = s * (x[1] - x[0])
    y_dot[1] = r * x[0] - x[0] * x[2] - x[1]
    y_dot[2] = x[0] * x[1] - b * x[2]
    

@cython.boundscheck(False)
@cython.wraparound(False)
cdef lorenz_step(
    double *x_i,
    double *x_f,
    double h,
    double s, double b, double r
):
    """ Implements a fourth order Runge Kutta step using vector_field().

    Args:
        x_i: Pointer to initial position
        x_f: Pointer to final position
        h: Time step
        t: Time. Not used.
        s: Lorenz parameter
        b: Lorenz parameter
        r: Lorenz parameter

    """
    cdef double k[5][3] # Storage for intermediate results.

    vector_field(x_i,s,b,r,k[0])
    for i in range(3):
        k[1][i] = h*k[0][i]
        x_f[i] = x_i[i] + k[1][i]/2

    vector_field(x_f,s,b,r,k[0])
    for i in range(3):
        k[2][i] = h*k[0][i]
        x_f[i] = x_i[i] + k[2][i]/2

    vector_field(x_f,s,b,r,k[0])
    for i in range(3):
        k[3][i] = h*k[0][i]
        x_f[i] = x_i[i] + k[3][i]

    vector_field(x_f,s,b,r,k[0])
    for i in range(3):
        k[4][i] = h*k[0][i]
        x_f[i] = x_i[i] + (k[1][i] + 2 * k[2][i] + 2 * k[3][i] + k[4][i])/6
    
@cython.boundscheck(False)
cdef lorenz_n_steps(
    double *x_i,
    double *x_f,
    double h,
    double s,
    double b,
    double r,
    int n,
):

    cdef double buffer[6]  # Double buffer for calculations
    cdef double *p_initial = &buffer[0]
    cdef double *p_final = & buffer[3]
    cdef double *p_temp
    cdef int i

    # Copy numpy initial condition into double buffer
    for i in range(3):
        p_initial[i] = x_i[i]

    # Do the n iterations
    for i in range(n):
        lorenz_step(p_initial, p_final, h, s, b, r)
        p_temp = p_initial
        p_initial = p_final
        p_final = p_temp

    # Copy the result into the numpy array
    for i in range(3):
        x_f[i] = p_initial[i]
    

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tangent_field(double *x, double s, double b, double r,
             double *y_dot):
    """This function acts as if it were pure C.  It calculates the tangent
    field (y_dot) at x,t for the Lorenz system with parmaters s, b, r.

    """
    cdef double df_dx[9]

    df_dx[0] = -s        # 0,0
    df_dx[1] = s         # 0,1
    df_dx[2] = 0         # 0,2
    df_dx[3] = r - x[2]  # 1,0
    df_dx[4] = -1        # 1,1
    df_dx[5] = -x[0]     # 1,2
    df_dx[6] = x[1]      # 2,0
    df_dx[7] = x[0]      # 2,1
    df_dx[8] = -b        # 2,2
    
    y_dot[0] = s * (x[1] - x[0])
    y_dot[1] = r * x[0] - x[0] * x[2] - x[1]
    y_dot[2] = x[0] * x[1] - b * x[2]
    
    y_dot[3] = df_dx[0] * x[3] + df_dx[1] * x[4] + df_dx[2] * x[5]
    y_dot[4] = df_dx[3] * x[3] + df_dx[4] * x[4] + df_dx[5] * x[5]
    y_dot[5] = df_dx[6] * x[3] + df_dx[7] * x[4] + df_dx[8] * x[5]
    
    y_dot[6] = df_dx[0] * x[6] + df_dx[1] * x[7] + df_dx[2] * x[8]
    y_dot[7] = df_dx[3] * x[6] + df_dx[4] * x[7] + df_dx[5] * x[8]
    y_dot[8] = df_dx[6] * x[6] + df_dx[7] * x[7] + df_dx[8] * x[8]
    
    y_dot[9] = df_dx[0] * x[9] + df_dx[1] * x[10] + df_dx[2] * x[11]
    y_dot[10]= df_dx[3] * x[9] + df_dx[4] * x[10] + df_dx[5] * x[11]
    y_dot[11]= df_dx[6] * x[9] + df_dx[7] * x[10] + df_dx[8] * x[11]
    

@cython.boundscheck(False)
cdef tangent_step(
    double *x_i,
    double *x_f,
    double h,
    double s, double b, double r
):
    """ Implements a fourth order Runge Kutta step using vector_field().

    Args:
        x_i: Pointer to initial tangent
        x_f: Pointer to final tangent
        h: Time step
        t: Time. Not used.
        s: Lorenz parameter
        b: Lorenz parameter
        r: Lorenz parameter

    """
    cdef double k[5][12] # Storage for intermediate results.

    tangent_field(x_i,s,b,r,k[0])
    for i in range(12):
        k[1][i] = h*k[0][i]
        x_f[i] = x_i[i] + k[1][i]/2

    tangent_field(x_f,s,b,r,k[0])
    for i in range(12):
        k[2][i] = h*k[0][i]
        x_f[i] = x_i[i] + k[2][i]/2

    tangent_field(x_f,s,b,r,k[0])
    for i in range(12):
        k[3][i] = h*k[0][i]
        x_f[i] = x_i[i] + k[3][i]

    tangent_field(x_f,s,b,r,k[0])
    for i in range(12):
        k[4][i] = h*k[0][i]
        x_f[i] = x_i[i] + (k[1][i] + 2 * k[2][i] + 2 * k[3][i] + k[4][i])/6
    

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tangent_n_steps(
    DTYPE_t *x_i,
    DTYPE_t *x_f,
    double h,
    double s, double b, double r,
    int n,
):

    cdef double buffer[24]
    cdef double *p_initial = &buffer[0]
    cdef double *p_final = & buffer[12]
    cdef double *p_temp
    cdef int i

    # Copy intial condition into calculation double buffer
    for i in range(12):
        p_initial[i] = x_i[i]
        
    for i in range(n):
        tangent_step(p_initial, p_final, h, s, b, r)
        p_temp = p_initial
        p_initial = p_final
        p_final = p_temp

    # Copy result into numpy result
    for i in range(12):
        x_f[i] = p_initial[i]
    

import hmm.state_space
import scipy.integrate
import typing

class SDE(hmm.state_space.SDE):
    """Reimplements methods that integrate the ODE.  Namely: forecast and
    simulate.

    """
    
    def simulate(self, x_initial: numpy.ndarray, t_initial: float,
                 t_final: float):
        """Integrate ODE, add noise, and observe with noise.

        Args:
            x_initial:
            t_initial:
            t_final:

        Returns:
            (x_final, y_final)
        """

        state_noise = numpy.sqrt(t_final - t_initial) * numpy.dot(
            self.unit_state_noise, self.rng.standard_normal(self.x_dim))

        x_final = lorenz_integrate(x_initial, t_initial, t_final)

        observation = self.observation_function(t_final, x_final)[0]
        observation_noise = numpy.dot(
            self.observation_noise_multiplier,
            self.rng.standard_normal(len(observation)))

        return x_final, observation + observation_noise

    
    def forecast(self, x_initial: numpy.ndarray, t_initial: float,
                 t_final: float):
        """Calculate parameters for forecast step of Kalman filter

        Args:
            x_initial: State at time t_inital
            t_initial:
            t_final:

        Returns:
            (x_final, d x_final/d x_initial, state_covariance)

        """

        #assert x_initial.shape == (self.x_dim,)
        # t_initial > t_final if used as backcast
        abs_dt = abs(t_final - t_initial)

        state_noise_covariance = (abs_dt * numpy.dot(
            self.unit_state_noise, self.unit_state_noise.T)) * self.fudge

        x_aug = numpy.empty((1 + self.x_dim) * self.x_dim)
        x_aug[:self.x_dim] = x_initial
        x_aug[self.x_dim:] = numpy.eye(self.x_dim).reshape(-1)

        result_aug = tangent_integrate(x_aug, t_initial, t_final)
        x_final = result_aug[:self.x_dim]
        derivative = result_aug[self.x_dim:].reshape(self.x_dim, self.x_dim)
        return x_final, derivative, state_noise_covariance
 
#--------------------------------
# Local Variables:
# mode: python
# End:
