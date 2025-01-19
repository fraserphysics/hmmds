r"""utilities.py

The functions in this module may be imported into other scripts to
provide a python interface to tools for integrating the lorenz system.


.. math::
    \dot x = s(y-x)

    \dot y = rx -xz -y

    \dot z = xy -bz

"""

import numpy
import scipy.integrate  # type: ignore


def dx_dt(t, x, s=10.0, r=28.0, b=8.0 / 3):
    """Lorenz vector field at x for scipy.integrate.solve_ivp
    Args:
        t: Not used, but set in call from solve_ivp
        x: 3-d state of Lorenz system
        s,r,b: Parameters of Lorenz system
    """
    return numpy.array([
        s * (x[1] - x[0]),  #
        x[0] * (r - x[2]) - x[1],  #
        x[0] * x[1] - b * x[2]
    ])


def tangent(t, x_dx, s, r, b):
    """Lorenz vector field and tangent for scipy.integrate.solve_ivp
    
    Args:
        t: Not used, but set in call from solve_ivp
        x_dx: Flattened combination of 3-d state and 3x3 Jacobian
        s,r,b: Parameters of Lorenz system

    """
    result = numpy.empty(12)  # Allocate storage for result
    assert x_dx.shape == result.shape

    # Unpack state and derivative from argument
    x = x_dx[:3]
    dx_dx0 = x_dx[3:].reshape((3, 3))

    # First three components are the value of the vector field F(x)
    result[:3] = dx_dt(t, x, s, r, b)

    dF = numpy.array([  # The derivative of F wrt x
        [-s, s, 0],  #
        [r - x[2], -1, -x[0]],  #
        [x[1], x[0], -b]
    ])

    # Assign the tangent part of the return value.
    result[3:] = numpy.dot(dF, dx_dx0).reshape(-1)

    return result


def integrate_tangent(t, x, jacobian, atol=1e-7):
    """
    Args:
        t: integration time
        x: 3-d state
        jacobian: 3x3 derivative
    """
    s = 10.0
    r = 28.0
    b = 8.0 / 3
    method = 'RK45'
    augmented_state = numpy.concatenate((x, jacobian.flatten()))
    bunch = scipy.integrate.solve_ivp(tangent, (0.0, t),
                                      augmented_state,
                                      args=(s, r, b),
                                      atol=atol,
                                      method=method)
    assert bunch.success, f'{bunch}'
    new_x = bunch.y[:3, -1]
    new_tangent = bunch.y[3:, -1]
    return new_x, new_tangent.reshape((3, 3))


def get_bins(args):
    """Boundaries for quantization.

    In a function so that other modules can access.
    """
    return numpy.linspace(-20, 20, args.levels + 1)[1:-1]



class FixedPoint:
    """Characterizes a focus of the Lorenz system

    Args:
        r,s,b: Parameters of Lorenz system
        sign: Specifies which focus to characterize
    """

    def __init__(
            self,  # FixedPoint
            r=28.0,
            s=10.0,
            b=8.0 / 3,
            t_sample=0.15,
            sign=1,
    ):
        self.r = r
        self.s = s
        self.b = b
        self.t_sample = t_sample
        if sign == 0:
            self.fixed_point = numpy.zeros(3)
            return
        assert abs(sign) == 1
        root = sign*numpy.sqrt(b * (r - 1))
        self.fixed_point = numpy.array([root, root, r - 1])

    def dPhi_dx(self, t_sample):
        """ Intgrate and return tangent
        """
        _, result = integrate_tangent(t_sample, self.fixed_point, numpy.eye(3))
        return result

#---------------
# Local Variables:
# mode:python
# End:
