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
    parser.add_argument("--reference_model",
                        type=str,
                        help="Path to model for calculating thresholds")
    parser.add_argument(
        'key',
        type=str,
        help='One of the functions registered in the source, eg, apnea_dict')
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def dict2hmm(state_dict, make_observation_model, args, rng):
    """Create an HMM based on state_dict for supervised training

    Args:
        state_dict: state_dict[state_key] is a State instance
        make_observation_model: Function
        args: 
        rng: A random number generator

    Return: hmm, state_key2state_index

    """

    n_states = len(state_dict)
    p_state_initial = numpy.ones(n_states) / n_states
    p_state_time_average = numpy.ones(n_states) / n_states
    p_state2state = hmm.simple.Prob(numpy.zeros((n_states, n_states)))
    state_key2state_index = {}
    state_keys = []
    untrainable_indices = []
    untrainable_values = []

    # Build state_key2state_index, p_state2state and state_keys
    for state_index, state_key in enumerate(state_dict.keys()):
        state_key2state_index[state_key] = state_index
        state_keys.append(state_key)
    for state_key in state_keys:
        state = state_dict[state_key]
        state_index = state_key2state_index[state_key]
        for successor_key, probability, trainable in zip(
                state.successors, state.probabilities, state.trainable):
            successor_index = state_key2state_index[successor_key]
            p_state2state[state_index, successor_index] = probability
            if not trainable:
                untrainable_indices.append((state_index, successor_index))
                untrainable_values.append(probability)
    p_state2state.normalize()

    # Create and return the hmm
    return develop.HMM(p_state_initial,
                       p_state_time_average,
                       p_state2state,
                       make_observation_model(state_dict, state_keys, rng,
                                              args),
                       args,
                       rng,
                       untrainable_indices=tuple(
                           numpy.array(untrainable_indices).T),
                       untrainable_values=numpy.array(
                           untrainable_values)), state_key2state_index


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


def make_varg(state_dict: dict, keys: list, rng,
              args) -> hmm.base.JointObservation:
    """Return a JointObservation instance with components
    "hr_respiration" and "class" or "threshold"

    Args:
        state_dict: Parameters for s in state_dict[s].observation
        keys: Establishes order for state_dict
        rng: numpy random number generator
        args: Command line arguments

    Return: result with

    result["hr_respiration"] is a VARG instance
    option: result['class'] is a hmm.base.ClassObservation instance
    option: result['threshold'] is a hmm.observe_float.GaussMAP instance

    """

    n_states = len(keys)
    assert n_states == len(state_dict)
    y_dim = 2
    ar_order = args.AR_order
    coefficient_shape = (n_states, y_dim, y_dim * ar_order + 1)
    # Mean of inverse Wishart without data is psi/nu

    ar_coefficients = numpy.zeros(coefficient_shape)

    sigma = numpy.empty((n_states, y_dim, y_dim))
    psi = numpy.empty((n_states, y_dim, y_dim))
    nu = numpy.empty(n_states)

    has_class = 'class' in state_dict[keys[0]].observation
    if has_class:
        class_index2state_indices = {0: [], 1: []}

    has_threshold = 'threshold' in state_dict[keys[0]].observation
    if has_threshold:
        mu = numpy.empty(n_states)
        variance = numpy.empty(n_states)
        alpha = numpy.empty(n_states)
        beta = numpy.empty(n_states)

    for state_index, (key, parameters) in enumerate(
        (key, state_dict[key].observation) for key in keys):

        varg = parameters['hr_respiration']
        ar_coefficients[state_index] = varg['coefficients']
        sigma[state_index] = varg['sigma']
        psi[state_index] = varg['psi']
        nu[state_index] = varg['nu']

        if has_class:
            class_index2state_indices[parameters['class']].append(state_index)

        if has_threshold:
            mu[state_index] = parameters['threshold']['mu']
            variance[state_index] = parameters['threshold']['variance']
            alpha[state_index] = parameters['threshold']['alpha']
            beta[state_index] = parameters['threshold']['beta']

    if has_class:
        return hmm.base.JointObservation(
            {
                'hr_respiration':
                    hmm.observe_float.VARG(
                        ar_coefficients, sigma, rng, Psi=psi, nu=nu),
                'class':
                    hmm.base.ClassObservation(class_index2state_indices),
            },
            power=args.power_dict)

    return hmm.base.JointObservation(
        {
            'hr_respiration':
                hmm.observe_float.VARG(
                    ar_coefficients, sigma, rng, Psi=psi, nu=nu)
        },
        power=args.power_dict)


@register  # VARG for low pass heart rate and derived respiration
def multi_state(args, rng):
    """Fully connected HMM with VARG models for respiration and heart rate

    """

    args.read_y_class = hmmds.applications.apnea.model_init.read_lphr_respiration_class
    args.read_raw_y = hmmds.applications.apnea.model_init.read_lphr_respiration
    observation_args = {  # Default values for apnea not noise
        'hr_respiration': {
            'sigma': numpy.eye(2) * 1.0e4,
            'psi': numpy.array([[0.4, 0.0], [0.0, 0.00006]]) * 1.0e4,
            'nu': 1.0e4
        },
        'class': 1,  # Default is apnea
    }

    small = 1.0e-30
    v_dim = 2  # Dimension of varg observation
    state_dict = {}
    state_keys = '''
    normal_noise
    normal_0
    normal_1
    apnea_noise
    apnea_0
    apnea_1
    '''.split()
    n_states = len(state_keys)
    for state_index, state_key in enumerate(state_keys):
        p_successors = numpy.ones(n_states) / (n_states - 2)
        trainable = [True] * n_states
        for successor_index, successor_key in enumerate(state_keys):
            if successor_key.find(
                    'noise') > 0 and successor_index != state_index:
                p_successors[successor_index] = small
                trainable[successor_index] = False

        # Observation parameters for state
        observation_copy = copy.deepcopy(observation_args)

        temp = numpy.zeros((v_dim, v_dim * args.AR_order + 1))
        temp[0, :] = rng.uniform(.8, 1.2)  # Break symmetry
        observation_copy['hr_respiration']['coefficients'] = temp

        if state_key.find('normal') >= 0:
            observation_copy['class'] = 0

        # psi/nu is covariance without data
        if state_key.find('noise') >= 0:
            observation_copy['hr_respiration']['sigma'] = numpy.eye(2) * 1.0e8
            observation_copy['hr_respiration']['psi'] = numpy.eye(2) * 1.0e6
            observation_copy['hr_respiration']['nu'] = 1.0e4
        state_dict[state_key] = State(state_keys,
                                      p_successors,
                                      observation_copy,
                                      trainable=trainable)

    result_hmm, state_key2state_index = dict2hmm(state_dict, make_varg, args,
                                                 rng)

    return result_hmm, state_dict, state_key2state_index


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
