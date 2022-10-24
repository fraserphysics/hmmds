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
                        default=1e-4,
                        help='For generating data')
    parser.add_argument('--dev_state',
                        type=float,
                        default=1e-6,
                        help='For generating data')
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
    atol = 1.0e-8
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
    covariance = made_dict['stationary_distribution'].covariance / 1e4
    initial_distribution = hmm.state_space.MultivariateNormal(
        made_dict['initial_state'], covariance, rng)
    return system, initial_distribution


def ekf_entropy(time_step, log_observation_noise, args, initial_distribution,
                y):

    dev_observation = 10**log_observation_noise
    filter_system, _ = make_system(dev_observation, args.dev_state, time_step,
                                   args.y_step)
    forecast_means, forecast_covariances, update_means, update_covariances, y_means, y_variances, y_probabilities = filter_system.forward_filter(
        initial_distribution, y)
    safety = 1e-10  #FixMe make this a Gaussian using y_means and y_covariances
    result = numpy.log(y_probabilities + safety).sum() / len(y)
    print(f'ts {time_step}, log_noise {log_observation_noise}, result {result}')
    assert result > -200
    return result


def survey(args):
    """Do a two dimensional survey cross entropy over sampling
    interval t_s and dev_observation, the scale of the observation
    noise.

    """
    dev_state = args.dev_state
    cross_entropy = {}
    for time_step in numpy.arange(*args.t_steps):
        generation_system, initial_distribution = make_system(
            args.dev_measurement, dev_state, time_step, args.y_step)
        x, y = generation_system.simulate_n_steps(initial_distribution,
                                                  args.n_t, args.y_step)
        cross_entropy[time_step] = {}
        for log_observation_noise in numpy.arange(*args.log_steps):
            cross_entropy[time_step][log_observation_noise] = ekf_entropy(
                time_step, log_observation_noise, args, initial_distribution, y)
    return cross_entropy


def main(argv=None):
    """Study dependence of relative entropy of Kalman filters on
    actual sample time and assumed observation noise.

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    cross_entropy = survey(args)
    for time_step in cross_entropy.keys():
        for log_observation_noise in cross_entropy[time_step].keys():
            print(
                f'{time_step} {log_observation_noise} {cross_entropy[time_step][log_observation_noise]}'
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:
