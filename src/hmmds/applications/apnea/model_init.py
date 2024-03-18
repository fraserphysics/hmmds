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
    small = 1e-30
    p_state2state = hmm.simple.Prob(numpy.array([[1.0, small], [1.0, small]]))
    untrainable_indices = []
    untrainable_values = []
    for from_state in range(n_states):
        for to_state in range(n_states):
            untrainable_indices.append((from_state, to_state))
            untrainable_values.append(p_state2state[from_state, to_state])

    class_index2state_indices = {0: [0], 1: [1]}
    state_key2state_index = {0: 0, 1: 1}

    v_dim = 2
    ar_order = args.AR_order
    coefficients = numpy.zeros((n_states, v_dim, v_dim * ar_order + 1))
    for state in range(n_states):
        coefficients[state, 0] = 1.0

    sigma = numpy.empty((n_states, v_dim, v_dim))
    Psi = numpy.empty((n_states, v_dim, v_dim))
    nu = numpy.empty(n_states)
    sigma_0 = numpy.array([[1.0, 0], [0, 1.0e-4]])
    _nu = 1.0

    for state in range(n_states):
        sigma[state, :, :] = sigma_0 * 1.0e8
        Psi[state, :, :] = numpy.array([[1.0e6, 0], [0, 1.0e3]]) * _nu
        nu[state] = _nu

    y_model = hmm.base.JointObservation({
        'hr_respiration':
            hmm.observe_float.VARG(coefficients, sigma, rng, Psi=Psi, nu=nu),
        'class':
            hmm.base.ClassObservation(class_index2state_indices),
    })

    state_dict = {0: State([0], [1.0], y_model), 1: State([0], [1.0], y_model)}

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


def make_varg(state_dict: dict, keys: list, rng,
              args) -> hmm.base.JointObservation:
    """Return a JointObservation instance with components "hr_respiration" and "class"

    Args:
        state_dict: Parameters for s in state_dict[s].observation
        keys: Establishes order for state_dict
        rng: numpy random number generator
        args: Command line arguments

    Return: result with

    result["hr_respiration"] is a VARG instance
    result['class'] is a hmm.base.ClassObservation instance

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

    for state_index, (key, parameters) in enumerate(
        (key, state_dict[key].observation) for key in keys):

        varg = parameters['hr_respiration']
        ar_coefficients[state_index] = varg['coefficients']
        sigma[state_index] = varg['sigma']
        psi[state_index] = varg['psi']
        nu[state_index] = varg['nu']

        if has_class:
            class_index2state_indices[parameters['class']].append(state_index)

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
    apnea_noise
    normal_0
    normal_1
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


@register  # Like multi_state, but for estimating threshold
def classless(args, rng):
    """Fully connected HMM with VARG models for respiration and heart rate

    """

    args.read_raw_y = hmmds.applications.apnea.model_init.read_lphr_respiration
    observation_args = {
        'hr_respiration': {
            'sigma': numpy.eye(2) * 1.0e4,
            'psi': numpy.array([[100.0, 0.0], [0.0, 2.0]]) * 1.0e2,
            'nu': 1.0e2
        }
    }

    v_dim = 2  # Dimension of varg observation
    state_dict = {}
    state_keys = '''state_0 state_1 state_2 state_3'''.split()
    n_states = len(state_keys)
    for state_key in state_keys:
        p_successors = numpy.ones(n_states) / n_states

        # Observation parameters for state
        observation_copy = copy.deepcopy(observation_args)
        temp = numpy.zeros((v_dim, v_dim * args.AR_order + 1))
        temp[0, :] = rng.uniform(.8, 1.2)  # Break symmetry
        observation_copy['hr_respiration']['coefficients'] = temp

        state_dict[state_key] = State(state_keys, p_successors,
                                      observation_copy)

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
