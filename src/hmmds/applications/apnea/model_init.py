"""model_init.py Create initial HMM models with apnea observations

"""
import sys
import os.path
import pickle
import argparse
import copy

import numpy

import hmm.base
import hmm.simple

import hmm.C
import develop
import utilities
from utilities import State
import hmmds.applications.apnea.model_init


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Create and write/pickle an initial model")
    utilities.common_arguments(parser)
    parser.add_argument("--alpha_beta",
                        type=float,
                        nargs=2,
                        default=(5.0e1, 1.0e1),
                        help="Paramters of inverse gamma prior for variance")
    parser.add_argument(
        'key',
        type=str,
        help='One of the functions registered in the source, eg, apnea_dict')
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


MODELS = {}  # Is populated by @register decorated functions.  The keys
# are function names, and the values are functions


def register(func):
    """Decorator that puts function in MODELS dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    MODELS[func.__name__] = func
    return func


@register
def test(args, rng):
    for key, value in args.__dict__.items():
        print(f'{key}: {value}')


def read_slow_class(args, record_name):
    heart_rate = utilities.HeartRate(args, record_name)
    heart_rate.filter_hr()
    heart_rate.read_expert()
    return heart_rate.dict('slow class'.split())


def read_slow(args, record_name):
    heart_rate = utilities.HeartRate(args, record_name)
    heart_rate.filter_hr()
    return heart_rate.dict(['slow'])


def read_lphr_respiration_class(args, record_name):
    """
    """

    keys = 'hr_respiration class'.split()
    item_args = {'hr_respiration': {'pad': args.AR_order}}

    hr_instance = utilities.HeartRate(args, record_name)
    hr_instance.read_expert()
    resp_pass_center = args.band_pass_center
    resp_pass_width = args.band_pass_width
    low_pass_width = 1 / args.low_pass_period
    hr_instance.filter_hr(resp_pass_center, resp_pass_width,
                          args.respiration_smooth, low_pass_width)

    return hr_instance.dict(keys, item_args)


def read_lphr_respiration(args, record_name):
    """
    """

    keys = ['hr_respiration']
    item_args = {'hr_respiration': {'pad': args.AR_order}}

    hr_instance = utilities.HeartRate(args, record_name)
    resp_pass_center = args.band_pass_center
    resp_pass_width = args.band_pass_width
    low_pass_width = 1 / args.low_pass_period
    hr_instance.filter_hr(resp_pass_center, resp_pass_width,
                          args.respiration_smooth, low_pass_width)

    return hr_instance.dict(keys, item_args)


@register  # Model for "c" records
def c_model(args, rng):
    """Return an hmm that finds all minutes normal

    """

    n_states = 2
    p_state_initial = numpy.array([1.0, 0])
    p_state_time_average = numpy.array([1.0, 0])
    p_state2state = hmm.simple.Prob(numpy.ones((n_states, n_states))) / 2

    class_index2state_indices = {0: [0], 1: [1]}
    state_key2state_index = {0: 0, 1: 1}

    y_model = hmm.base.JointObservation({
        'slow':
            hmm.observe_float.Gauss(numpy.array([50, -1e6]),
                                    numpy.ones(2) * 1.0e4, rng),
        'class':
            hmm.base.ClassObservation(class_index2state_indices),
    })

    state_dict = {0: State([0], [1.0], y_model), 1: State([0], [1.0], y_model)}

    # Create and return the hmm
    hmm_ = develop.HMM(p_state_initial, p_state_time_average, p_state2state,
                       y_model, args, rng)
    args.read_y_class = hmmds.applications.apnea.model_init.read_slow_class
    args.read_raw_y = hmmds.applications.apnea.model_init.read_slow
    return hmm_, state_dict, state_key2state_index


@register  # Has noise states
def four_state(args, rng):
    """Return an hmm with two states with VARG observation models.

    state 0: Typical normal (not apnea) state
    state 1: Normal noise state (to handle eg, lead noise)
    state 2: Typical apnea state
    state 3: Apnea noise state

    """

    normal_class = 0
    apnea_class = 1

    normal_state = 0
    normal_noise_state = 1
    apnea_state = 2
    apnea_noise_state = 3

    n_states = 4
    v_dim = 2
    small = 1.0e-50
    ar_order = args.AR_order
    p_state_initial = numpy.ones(n_states) / n_states
    p_state_time_average = p_state_initial.copy()
    p_state2state = hmm.simple.Prob(
        numpy.ones((n_states, n_states)) * small * small)
    p_state2state[normal_noise_state, normal_noise_state] = small
    p_state2state[apnea_noise_state, apnea_noise_state] = small

    for x, y in ((normal_state, normal_state), (normal_state, apnea_state),
                 (normal_noise_state, normal_state), (apnea_state, apnea_state),
                 (apnea_state, normal_state), (apnea_noise_state, apnea_state)):
        p_state2state[x, y] = 1.0
    p_state2state.normalize()

    class_index2state_indices = {
        normal_class: [normal_state, normal_noise_state],
        apnea_class: [apnea_state, apnea_noise_state]
    }

    coefficients = numpy.zeros((n_states, v_dim, v_dim * ar_order + 1))
    for state in range(n_states):
        coefficients[state, 0] = 1.0

    sigma = numpy.empty((n_states, v_dim, v_dim))
    Psi = numpy.empty((n_states, v_dim, v_dim))
    nu = numpy.empty(n_states)

    # I chose initial sigma to get convergence in 7 training
    # iterations.  I believe that the model structure is so simple
    # that the result of training to convergence gets a unique result
    # that is independent of variations in the initial model.
    sigma_0 = numpy.array([[1.0, 0], [0, 1.0e-4]])
    # Noise states visited about 200 times.  Regular states visited
    # about 25,000 times
    _nu = 500.0
    for state in (normal_state, apnea_state):
        sigma[state, :, :] = sigma_0 * 1.0e3
        Psi[state, :, :] = numpy.array([[50.0, 0], [0, .001]]) * _nu
        nu[state] = _nu

    for state in (normal_noise_state, apnea_noise_state):
        sigma[state, :, :] = sigma_0 * 1.0e8
        Psi[state, :, :] = numpy.array([[1.0e8, 0], [0, 1.0e4]]) * _nu
        nu[state] = _nu

    y_model = hmm.base.JointObservation({
        'hr_respiration':
            hmm.observe_float.VARG(coefficients, sigma, rng, Psi=Psi, nu=nu),
        'class':
            hmm.base.ClassObservation(class_index2state_indices),
    })

    untrainable_indices = []
    untrainable_values = []
    for from_state in range(n_states):
        for to_state in (normal_noise_state, apnea_noise_state):
            if from_state == to_state:
                continue
            untrainable_indices.append((from_state, to_state))
            untrainable_values.append(small)
    for from_state, to_state in ((normal_noise_state, apnea_state),
                                 (apnea_noise_state, normal_state)):
        untrainable_indices.append((from_state, to_state))
        untrainable_values.append(small)

    # Create and return the hmm
    hmm_ = develop.HMM(p_state_initial,
                       p_state_time_average,
                       p_state2state,
                       y_model,
                       args,
                       rng,
                       untrainable_indices=tuple(
                           numpy.array(untrainable_indices).T),
                       untrainable_values=numpy.array(untrainable_values))
    args.read_y_class = hmmds.applications.apnea.model_init.read_lphr_respiration_class
    args.read_raw_y = hmmds.applications.apnea.model_init.read_lphr_respiration

    # Next two lines are for debugging more complicated models
    state_key2state_index = {}
    state_dict = {}

    return hmm_, state_dict, state_key2state_index


def main(argv=None):
    """Create an hmm and write it as a pickle.
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    rng = numpy.random.default_rng(3)

    # Run the function specified by args.key
    model, state_dict, state_key2state_index = MODELS[args.key](args, rng)
    assert model.p_state_initial.min() >= 0

    model.strip()
    args.state_dict = state_dict
    args.state_key2state_index = state_key2state_index
    with open(args.write_path, 'wb') as _file:
        pickle.dump(model, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
