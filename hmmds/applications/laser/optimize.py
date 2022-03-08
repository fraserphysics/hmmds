"""optimize.py Find parameters of extended Kalman filter for laser data.

Here is the result for fitting the first 500 laser points with Nelder-Mead:

parameters_min:
delta_t 0.034536936493082204
t_ratio 0.9957172327687029
x_ratio 0.705753598217994
offset 14.902902478718389
state_noise 0.6869357939046703
observation_noise 0.5122634297815053

f(x_min)=-1103.7056771377574
success=True
message=Optimization terminated successfully.
iterations=720

Powell on 2876:
parameters_min:
delta_t 0.034864197628657737
t_ratio 0.9971526009958418
x_ratio 0.707111155695998
offset 14.917017176643578
state_noise 0.7021345705781504
observation_noise 0.5625303823620377

f(x_min)=-6447.25882156127
success=True
message=Optimization terminated successfully.
iterations=2

real	4m2.290s
user	4m3.718s
sys	0m0.165s

"""
import sys
import typing
import argparse

import numpy
import scipy.optimize  # minimize( function, x_0, method='BFGS')

import hmm.state_space
import hmmds.synthetic.filter.lorenz_sde

import explore
import plotscripts.introduction.laser


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Optimize parameters for laser data')
    parser.add_argument('--data',
                        type=str,
                        default='test_optimize',
                        help='Path to store data')
    return parser.parse_args(argv)


class Parameters:
    """Parameters for laser data.  Class associates names with values.
    """

    def __init__(
        self,
            delta_t = 0.034864197628657737,
            t_ratio = 0.9971526009958418,
            x_ratio = 0.707111155695998,
            offset = 14.917017176643578,
            state_noise = 0.7021345705781504,
            observation_noise = 0.5625303823620377,
        delta_x=2.993,
        fudge=1.0,
        laser_dt=0.04,
        s=10.0,
        r=30.0,
        b=8.0 / 3,
    ):
        self.delta_t = delta_t
        self.t_ratio = t_ratio
        self.x_ratio = x_ratio
        self.offset = offset
        self.state_noise = state_noise
        self.fudge = fudge  # Roll into state_noise
        self.observation_noise = observation_noise

        self.delta_x = delta_x
        self.laser_dt = laser_dt
        self.s = s
        self.r = r
        self.b = b
        self.variables = """
delta_t t_ratio x_ratio offset state_noise observation_noise delta_x""".split()

    def values(self):
        return tuple(getattr(self, key) for key in self.variables)

    def __str__(self):
        result = ''
        for key in self.variables:
            result += f'{key} {getattr(self,key)}\n'
        return result


# Global for access by objective_function
LASER_DATA = None


def objective_function(parameters_in, delta_x=None):
    """For optimization"""
    parameter = Parameters(*parameters_in)
    if delta_x is not None:
        parameter.delta_x=delta_x
    non_stationary, initial_distribution, initial_state = make_non_stationary(
        parameter, None)
    result = non_stationary.log_likelihood(initial_distribution, LASER_DATA)
#    print(f"""at
#{parameter}
#objective_function = {result}""")
    return -result


def make_non_stationary(args, rng):
    """Make an SDE system instance

    Args:
        args: Command line arguments
        rng:

    Returns:
        (An SDE instance, an initial state, an inital distribution)

    The goal is to get linear_map_simulation.main to exercise all of the
    SDE methods on the Lorenz system.

    """

    # The next three functions are passed to SDE.__init__

    def dx_dt(t, x, s, r, b):
        return numpy.array([
            s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]
        ])

    def tangent(t, x_dx, s, r, b):
        result = numpy.empty(12)  # Allocate storage for result

        # Unpack state and derivative from argument
        x = x_dx[:3]
        dx_dx0 = x_dx[3:].reshape((3, 3))

        # First three components are the value of the vector field F(x)
        result[:3] = dx_dt(t, x)

        dF = numpy.array([  # The derivative of F wrt x
            [-s, s, 0], [r - x[2], -1, -x[0]], [x[1], x[0], -b]
        ])

        # Assign the tangent part of the return value.
        result[3:] = numpy.dot(dF, dx_dx0).reshape(-1)

        return result

    def observation_function(
            t, state) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """Calculate observation and its derivative
        """
        y = numpy.array([args.x_ratio * state[0]**2 + args.offset])
        dy_dx = numpy.array([[args.x_ratio * 2 * state[0], 0, 0]])
        return y, dy_dx

    fixed_point = explore.FixedPoint(args.r)
    x_dim = 3
    state_noise = numpy.ones(x_dim) * args.state_noise
    y_dim = 1
    observation_noise = numpy.eye(y_dim) * args.observation_noise

    # lorenz_sde.SDE only uses Cython for methods forecast and simulate
    dt = args.laser_dt * args.t_ratio
    system = hmmds.synthetic.filter.lorenz_sde.SDE(dx_dt,
                                                   tangent,
                                                   state_noise,
                                                   observation_function,
                                                   observation_noise,
                                                   dt,
                                                   x_dim,
                                                   ivp_args=(args.s, args.r,
                                                             args.b),
                                                   fudge=args.fudge)
    x_0 = fixed_point.initial_state(args.delta_x)
    initial_mean = system.simulate(x_0, 0.0, args.delta_t)[0]
    initial_covariance = numpy.outer(state_noise, state_noise)
    initial_distribution = hmm.state_space.MultivariateNormal(
        initial_mean, initial_covariance)
    result = hmm.state_space.NonStationary(system, dt, rng)
    return result, initial_distribution, initial_mean

def study_delta_x():
    global LASER_DATA
    
    laser_data = plotscripts.introduction.laser.read_data('LP5.DAT')
    assert laser_data.shape == (2, 2876)
    length = 175
    LASER_DATA = laser_data[1, :length].astype(int).reshape((length, 1))

    defaults = Parameters().values()
    delta_x_array = numpy.linspace(1.895, 4.0, 50)
    result = numpy.empty(delta_x_array.shape)
    
    for i, delta_x in enumerate(delta_x_array):
        result[i] = -objective_function(defaults, delta_x)
    print(result)
    return delta_x_array, result

def optimize():
    global LASER_DATA

    laser_data = plotscripts.introduction.laser.read_data('LP5.DAT')
    assert laser_data.shape == (2, 2876)
    length = 2876
    LASER_DATA = laser_data[1, :length].astype(int).reshape((length, 1))

    parameters = Parameters()
    defaults = parameters.values()
    result = scipy.optimize.minimize(objective_function,
                                     defaults,
                                     #method='BFGS')
                                     method='Powell')
    parameters_max = Parameters(*result.x)
    print(f"""parameters_max:
{parameters_max}
f_max={-result.fun}
success={result.success}
message={result.message}
iterations={result.nit}""")

    
def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    #args = parse_args(argv)
    optimize()
    return 0
    delta_x, log_like = study_delta_x()
    import matplotlib.pyplot as plt
    plt.plot(delta_x, log_like)
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:
