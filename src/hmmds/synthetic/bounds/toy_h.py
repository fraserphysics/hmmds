"""toy_h.py Cross entropy vs sample time and measurement noise

"""
from __future__ import annotations  # Enables, eg, (self: LocalNonStationary

import sys
import argparse
import os
import typing
import pickle

import numpy
import numpy.linalg
import scipy.linalg
import scipy.special

import hmm.state_space
import hmmds.synthetic.filter.lorenz_sde
import hmmds.synthetic.bounds.lorenz


def parse_args(argv):
    """Parse a command line.
    """
    parser = argparse.ArgumentParser(
        description='Make survey of cross entropy vs t_s and observation noise')
    parser.add_argument('--n_t',
                        type=int,
                        default=2000,
                        help='Number of x and y samples')
    parser.add_argument('--dev_measurement',
                        type=float,
                        default=1e-10,
                        help='For generating data')
    parser.add_argument('--dev_state_generate',
                        type=float,
                        default=1e-6,
                        help='For generating data')
    parser.add_argument('--dev_state_filter',
                        type=float,
                        default=1e-6,
                        help='For EKF')
    parser.add_argument('--y_step',
                        type=float,
                        default=1e-4,
                        help='Quantization size')
    parser.add_argument('--t_steps',
                        type=float,
                        default=[.02, .51, .02],
                        nargs=3,
                        help='min, max, step_size for sample time')
    parser.add_argument('--log_steps',
                        type=float,
                        default=[-3.5, -5.6, -.1],
                        nargs=3,
                        help='min, max, step_size for log_10 noise')
    parser.add_argument('h_survey_path', type=str, help='path for result')
    parser.add_argument('h_tau_path', type=str, help='path for result')
    return parser.parse_args(argv)


def make_system(dev_observation, dev_state, time_step, y_step):
    """ Call hmmds.synthetic.bounds.lorenz.make_system and create an
    initial distribution.

    """
    s = 10.0
    r = 28.0
    b = 8.0 / 3
    h_max = 1.0e-3
    atol = 1.0e-7
    fudge = 1.0

    rng = numpy.random.default_rng(3)
    # In state_space.SDE.forecast, the covariance ends up being dt
    # * state_noise_scale**2.  So dividing by sqrt(dt) here makes
    # self.dev_state the actual noise scale.
    state_noise_scale = dev_state / numpy.sqrt(time_step)
    made_dict = hmmds.synthetic.bounds.lorenz.make_system(
        s, r, b, state_noise_scale, dev_observation, time_step, y_step,
        fudge**2, h_max, atol, rng)
    assert set(made_dict.keys()) == set(
        'Cython SciPy stationary_distribution initial_state'.split())
    system = made_dict['SciPy']

    # This initial_distribution is only used to generate data.
    # Filtering uses numpy.eye(3)*1e-3
    covariance = made_dict['stationary_distribution'].covariance / 1e-4
    initial_distribution = hmm.state_space.MultivariateNormal(
        made_dict['initial_state'], covariance, rng)
    return system, initial_distribution, rng


def cumulative(z):
    """Calculate the cumulative value for z in N(0,1)

    """
    return (1 + scipy.special.erf(z / numpy.sqrt(2))) / 2


def pmf(y, mean, dev, step):
    """Calculate the probability of the interval y +/- step/2
    
    """
    z_0 = (y - mean - step / 2) / dev
    z_1 = (y - mean + step / 2) / dev
    return cumulative(z_1) - cumulative(z_0)


def ekf_entropy(time_step: float, log_observation_noise: float, args,
                initial_distribution, y: numpy.ndarray) -> float:
    """
    Estimate the cross entropy of an extended Kalman filter.

    Args:
        time_step: Time between samples
        log_observation_noise: Log_10 of standard deviation of noise.
        args: Command line arguments
        initial_distribution:
        y: The time series of observations

    """

    dev_observation = 10**log_observation_noise
    filter_system, _, _ = make_system(dev_observation, args.dev_state_filter,
                                      time_step, args.y_step)
    _, _, _, _, y_means, y_variances, y_probabilities = filter_system.forward_filter(
        initial_distribution, y)
    # Check that probability mass calculation here matches calculation
    # in lorenz.py
    pmfs = pmf(y[:, 0], y_means, numpy.sqrt(y_variances), filter_system.y_step)
    assert numpy.allclose(pmfs, y_probabilities)
    # For second component of Gaussian mixture forecast probabilities
    # use y_means and deviation = 20.  Weight the second component by
    # 1.0-3.
    dev = 20.0
    weight = 1.0e-3
    safety = pmf(y[:, 0], y_means, dev, filter_system.y_step)
    p_y = (1 - weight) * pmfs + weight * safety
    assert len(p_y) == len(y)
    result = numpy.log(p_y).sum() / len(y)
    assert result > -4
    return result


def survey(args):
    """Do a two dimensional survey of cross entropy over the sampling
    interval, t_s, and the scale of the observation noise,
    dev_observation.

    """
    cross_entropy = {}
    for time_step in numpy.arange(*args.t_steps):
        generation_system, initial_for_generation, rng = make_system(
            args.dev_measurement, args.dev_state_generate, time_step,
            args.y_step)
        x, y = generation_system.simulate_n_steps(initial_for_generation,
                                                  args.n_t)
        initial_for_filter = hmm.state_space.MultivariateNormal(
            x[0],
            numpy.eye(3) * 1e-3, rng)
        cross_entropy[time_step] = {}
        for log_observation_noise in numpy.arange(*args.log_steps):
            cross_entropy[time_step][log_observation_noise] = ekf_entropy(
                time_step, log_observation_noise, args, initial_for_filter, y)
    return cross_entropy


def main(argv=None):
    """Study dependence of relative entropy of Kalman filters on
    actual sample time and assumed observation noise.

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    cross_entropy = survey(args)
    for time_step in sorted(cross_entropy.keys()):
        for log_observation_noise in sorted(cross_entropy[time_step].keys()):
            print(
                f'{time_step:6.3f} {log_observation_noise:7.3f} {cross_entropy[time_step][log_observation_noise]:7.3f}'
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
