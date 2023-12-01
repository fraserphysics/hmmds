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
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument(
        'key',
        type=str,
        help='One of the functions registered in the source, eg, apnea_dict')
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    utilities.join_common(args)
    return args


def make_joint_slow_peak_interval_class(
    state_dict,
    keys,
    rng,
    args,
):
    """Return a JointObservation instance with components "slow",
    "peak", "interval", and "class"

    Args:
        state_dict: Parameters for s in state_dict[s].observation
        keys: Establishes order for state_dict
        rng: numpy random number generator
        args: Command line arguments

    Return: a JointObservation instance

    result["slow"] is an AutoRecressive instance
    result["peak"] is an IntegerObservation instance
    result['interval'][s] is a pdf for intervals
    result['class'] is a hmm.base.ClassObservation instance

    """
    class_index2state_indices = {}
    n_states = len(keys)
    assert n_states == len(state_dict)

    # Make "slow" component
    ar_order = len(state_dict[keys[0]].observation['slow']['coefficients'])
    ar_coefficients = numpy.empty((n_states, ar_order))
    offsets = numpy.empty(n_states)
    variances = numpy.empty(n_states)
    alphas = numpy.empty(n_states)
    betas = numpy.empty(n_states)
    for state_index, key in enumerate(keys):
        observation = state_dict[key].observation
        slow = observation['slow']
        _class = observation['class']

        if _class in class_index2state_indices:
            class_index2state_indices[_class].append(state_index)
        else:
            class_index2state_indices[_class] = [state_index]

        ar_coefficients[state_index] = slow['coefficients']
        offsets[state_index] = slow['offset']
        variances[state_index] = slow['variance']
        alphas[state_index] = slow['alpha']
        betas[state_index] = slow['beta']

    # Make py_state for "peak" component
    peak_dimension = len(state_dict[keys[0]].observation['peak'])
    py_state = numpy.empty((n_states, peak_dimension))
    for state_index, key in enumerate(keys):
        py_state[state_index, :] = state_dict[key].observation['peak']

    power = dict(zip('slow peak interval class'.split(), args.power))
    return hmm.base.JointObservation(
        {
            'slow':
                hmm.C.AutoRegressive(ar_coefficients, offsets, variances, rng,
                                     alphas, betas),
            'peak':
                hmm.C.IntegerObservation(py_state, rng),
            'interval':
                utilities.IntervalObservation(
                    tuple(state_dict[key].observation['interval']
                          for key in keys),
                    args,
                ),
            'class':
                hmm.base.ClassObservation(class_index2state_indices),
        },
        power=power)


def make_joint_varg_peak_interval_class(
    state_dict: dict,
    keys: list,
    rng,
    args,
) -> hmm.base.JointObservation:
    """Return a JointObservation instance with components "hr_respiration",
    "peak", "interval", and "class"

    Args:
        state_dict: Parameters for s in state_dict[s].observation
        keys: Establishes order for state_dict
        rng: numpy random number generator
        args: Command line arguments

    Return: result with

    result["hr_respiration"] is a VARG instance
    result["peak"] is an IntegerObservation instance
    result['interval'][s] is a pdf for intervals
    result['class'] is a hmm.base.ClassObservation instance

    """

    n_states = len(keys)
    assert n_states == len(state_dict)
    y_dim, len_coefficients = state_dict[
        keys[0]].observation['hr_respiration']['coefficients'].shape

    # Arrays for "hr_respiration" component
    ar_coefficients = numpy.empty((n_states, y_dim, len_coefficients))
    sigma = numpy.empty((n_states, y_dim, y_dim))
    psi = numpy.empty((n_states, y_dim, y_dim))
    nu = numpy.empty(n_states)

    p_peak_state = numpy.empty((n_states, 2))

    interval_pdfs = []

    class_index2state_indices = {0: [], 1: []}

    for state_index, (key, parameters) in enumerate(
        (key, state_dict[key].observation) for key in keys):

        varg = parameters['hr_respiration']
        ar_coefficients[state_index] = varg['coefficients']
        sigma[state_index] = varg['sigma']
        psi[state_index] = varg['psi']
        nu[state_index] = varg['nu']

        p_peak_state[state_index, :] = parameters['peak']

        interval_pdfs.append(parameters['interval'])

        class_index2state_indices[parameters['class']].append(state_index)

    power = dict(zip('hr_respiration peak interval class'.split(), args.power))
    return hmm.base.JointObservation(
        {
            'hr_respiration':
                hmm.observe_float.VARG(
                    ar_coefficients, sigma, rng, Psi=psi, nu=nu),
            'peak':
                hmm.C.IntegerObservation(p_peak_state, rng),
            'interval':
                utilities.IntervalObservation(interval_pdfs, args),
            'class':
                hmm.base.ClassObservation(class_index2state_indices),
        },
        power=power)


def dict2hmm(state_dict, make_observation_model, args, rng):
    """Create an HMM based on state_dict for supervised training

    Args:
        state_dict: state_dict[state_key] is a State instance
        make_observation_model: Function
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
    heart_rate = utilities.HeartRate(args, record_name, args.config)
    heart_rate.filter_hr()
    heart_rate.read_expert()
    return heart_rate.dict('slow class'.split())


def read_slow(args, record_name):
    heart_rate = utilities.HeartRate(args, record_name, args.config)
    heart_rate.filter_hr()
    return heart_rate.dict(['slow'])


def read_lphr_respiration_class(args, record_name):
    """
    """

    keys = 'hr_respiration class'.split()
    item_args = {'hr_respiration': {'pad': args.AR_order}}

    # develop.HMM.read_y_with_class calls this with self.args, and
    # apnea_train.main wraps the result in hmm.base.JointSegment

    assert args.config.normalize == args.normalize

    hr_instance = utilities.HeartRate(args, record_name, args.config,
                                      args.normalize)
    hr_instance.read_expert()
    resp_pass_center = args.band_pass_center
    resp_pass_width = args.band_pass_width
    envelope_smooth = args.envelope_smooth
    low_pass_width = 1 / args.low_pass_period
    hr_instance.filter_hr(resp_pass_center, resp_pass_width, envelope_smooth,
                          low_pass_width)

    return hr_instance.dict(keys, item_args)


def read_lphr_respiration(args, record_name):
    """
    """

    keys = ['hr_respiration']
    item_args = {'hr_respiration': {'pad': args.AR_order}}

    # develop.HMM.read_y_with_class calls this with self.args, and
    # apnea_train.main wraps the result in hmm.base.JointSegment

    assert args.config.normalize == args.normalize

    hr_instance = utilities.HeartRate(args, record_name, args.config,
                                      args.normalize)
    resp_pass_center = args.band_pass_center
    resp_pass_width = args.band_pass_width
    envelope_smooth = args.envelope_smooth
    low_pass_width = 1 / args.low_pass_period
    hr_instance.filter_hr(resp_pass_center, resp_pass_width, envelope_smooth,
                          low_pass_width)

    return hr_instance.dict(keys, item_args)


def read_slow_peak_interval_class(args, record_name):
    """Called by HMM, and returns a
    dict of observation components

    Args:
        args: From HMM.args
        record_name: EG, 'a01'

    Components are slow, peak, interval and class.

    """
    keys = 'slow peak interval class'.split()
    item_args = {'slow': {'pad': args.AR_order}}

    # develop.HMM.read_y_with_class calls this with self.args, and
    # apnea_train.main wraps the result in hmm.base.JointSegment

    assert args.config.normalize == args.normalize

    hr_instance = utilities.HeartRate(args, record_name, args.config,
                                      args.normalize)
    hr_instance.read_expert()
    hr_instance.filter_hr()
    hr_instance.find_peaks()

    return hr_instance.dict(keys, item_args)


def read_slow_peak_interval(args, record_name):
    """Called by HMM, and returns a
    dict of observation components

    Args:
        args: From HMM.args
        record_name: EG, 'a01'

    Components are slow, peak, and interval.

    """
    keys = 'slow peak interval'.split()
    item_args = {'slow': {'pad': args.AR_order}}

    # develop.HMM.read_y_with_class calls this with self.args, and
    # apnea_train.main wraps the result in hmm.base.JointSegment

    assert args.config.normalize == args.normalize

    hr_instance = utilities.HeartRate(args, record_name, args.config,
                                      args.normalize)
    hr_instance.filter_hr()
    hr_instance.find_peaks()

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


# int_class, pdf_interval, peak_prob
def peak_chain(switch_key: str,
               prefix: str,
               peak_args,
               non_peak_args,
               state_dict,
               length=9):
    """Add a sequence of states to state_dict

    Args:
        switch_key: Key of state that links these chains
        prefix: For state keys
        args:
        peak_args: Arguments for make_observation_model
        non_peak_args: Arguments for make_observation_model
        rng:
        state_dict:
        length: Number of samples in peak pattern

    """

    if length < 2:
        raise ValueError(f'For peak chain, length must be > 1, but {length=}.')
    for index in range(length + 1):
        state_key = f'{prefix}_{index}'
        next_key = f'{prefix}_{index+1}'
        if index == int(length / 2):
            state_dict[state_key] = State([next_key], [1.0], peak_args)
        else:
            state_dict[state_key] = State([next_key], [1.0], non_peak_args)

    # Repair transitions in slow and last states
    slow_key = f'{prefix}_0'
    first_key = f'{prefix}_1'
    last_key = f'{prefix}_{length}'
    state_dict[slow_key].set_transitions([switch_key, slow_key, first_key],
                                         [1.0e-20, .8, .2],
                                         trainable=(False, True, True))
    state_dict[last_key].set_transitions([slow_key], [1.0])


def make_switch_noise(int_class, chain_keys, state_dict, noise_args,
                      switch_args):
    """Make states for switching and for noise

    Args:
        args:
        int_class: Either 0 or 1
        chain_keys: Names of states that switch links to
        state_dict:
        noise_args:  Arguments for make_observation_model for noise state
        switch_args:  Arguments for make_observation_model for switch state
    """
    # Magic numbers
    noise_p = 1.0e-10  # Probability of transition to noise state
    # from self or either switch state
    switch_p = 1.0e-10  # Probability of transition between classes
    p_switch_self = 1.0e-20  # Probability of tranistion from
    # switch state to self
    if int_class == 0:
        switch_key = 'N_switch'
        noise_key = 'N_noise'
        other_switch = 'A_switch'
        other_noise = 'A_noise'
    elif int_class == 1:
        switch_key = 'A_switch'
        noise_key = 'A_noise'
        other_switch = 'N_switch'
        other_noise = 'N_noise'
    else:
        raise ValueError(f'{int_class=} not in [0,1]')

    # Specify noise state
    state_dict[noise_key] = State([noise_key, other_noise, switch_key],
                                  [noise_p, noise_p, 1.0 - 2 * noise_p],
                                  noise_args,
                                  trainable=(False, False, False))

    # Specify switch state
    successors = [switch_key, noise_key, other_switch, other_noise] + chain_keys
    probabilities = numpy.ones(len(successors)) / len(chain_keys)
    probabilities[0] = p_switch_self
    probabilities[1] = noise_p
    probabilities[2] = switch_p
    probabilities[3] = noise_p
    probabilities /= probabilities.sum()
    trainable = [True] * len(successors)
    trainable[0:4] = [False] * 4
    state_dict[switch_key] = State(successors,
                                   probabilities,
                                   switch_args,
                                   trainable=trainable)


def two_chain(args, rng, read_y_class, read_raw_y):
    """Make an hmm with one chain for normal and one chain for apnea

    """

    #  Normal Nodes  N Switch A Switch  Apnea nodes
    #
    # *************                     ************
    #              \                   /
    #               \________  _______/
    #                |      |--|     |
    #               /--------  -------\
    #              /                   \
    # *************                     ************

    args.read_y_class = read_y_class
    args.read_raw_y = read_raw_y

    # Define alpha and beta of inverse gamma for noise states.
    # There are about 12,000 minutes of data, 25 records * 480 minutes
    noise_alpha = 9.6e4
    noise_prior_variance = 10.0**2
    noise_beta = noise_alpha * noise_prior_variance

    args_alpha, args_beta = args.alpha_beta

    variance = 1.0e3
    coefficients = numpy.zeros(args.AR_order)
    coefficients[0] = 1.0
    observation_args = {
        'slow': {
            'coefficients': coefficients,
            'alpha': args_alpha,
            'beta': args_beta,
            'offset': 0.0,
            'variance': variance
        },
        'class': None,  # Set in copies
        'peak': numpy.array([1.0, 0.0]),
        'interval': None,  # Set in copies
    }

    state_dict = {}

    def make_one_chain(char_class, int_class, pdf_class):
        """
        """
        switch_key = f'{char_class}_switch'
        chain_key = f'{char_class}_chain'
        chain_0 = f'{char_class}_chain_0'
        peak_prob = numpy.array([0.0, 1.0])
        peak_prob[0] = 0

        peak_args = copy.deepcopy(observation_args)
        non_peak_args = copy.deepcopy(observation_args)
        noise_args = copy.deepcopy(observation_args)
        switch_args = copy.deepcopy(observation_args)

        for _args in (peak_args, non_peak_args, noise_args, switch_args):
            _args['class'] = int_class
            _args['interval'] = pdf_class
        peak_args['peak'] = numpy.array([0.0, 1.0])
        noise_args['peak'] = numpy.array([0.5, 0.5])
        noise_args['alpha'] = noise_alpha
        noise_args['beta'] = noise_beta

        peak_chain(
            switch_key,
            chain_key,
            peak_args,
            non_peak_args,
            state_dict,
        )

        make_switch_noise(int_class, [chain_0], state_dict, noise_args,
                          switch_args)

    make_one_chain('N', 0, args.config.normal_pdf)
    make_one_chain('A', 1, args.config.apnea_pdf)
    #for key, value in state_dict.items():
    #    print(f'{key} {value}')

    result_hmm, state_key2state_index = dict2hmm(
        state_dict, make_joint_slow_peak_interval_class, args, rng)
    return result_hmm, state_dict, state_key2state_index


@register
def two_intervals(args, rng):
    """Return an hmm with two chains and joint observations that
    include "slow", "peak" and "interval"

    """
    return two_chain(
        args, rng,
        hmmds.applications.apnea.model_init.read_slow_peak_interval_class,
        hmmds.applications.apnea.model_init.read_slow_peak_interval)


@register
def two_normalized(args, rng):
    """Return an hmm with two chains and joint observations with
    normalized heart rate as "slow", and "peak" and "interval".

    """

    # Functions in utilities normalize heart rate if "'norm_avg' in args"
    args.norm_avg = args.config.norm_avg
    return two_chain(
        args, rng,
        hmmds.applications.apnea.model_init.read_slow_peak_interval_class,
        hmmds.applications.apnea.model_init.read_slow_peak_interval)


@register  # Model that uses respiration signal and low pass heart
# rate.
def varg2state(args, rng):
    """Return an hmm with two states with VARG observation models.

    """

    n_states = 2
    y_dim = 2
    ar_order = args.AR_order
    p_state_initial = numpy.ones(n_states) / n_states
    p_state_time_average = p_state_initial.copy()
    p_state2state = hmm.simple.Prob(numpy.ones((n_states, n_states))) / 2

    class_index2state_indices = {0: [0], 1: [1]}

    a = numpy.ones((n_states, y_dim, y_dim * ar_order + 1))
    sigma = numpy.empty((n_states, y_dim, y_dim))
    for state in range(n_states):
        sigma[state, :, :] = numpy.eye(2) * 1e6
    Psi = numpy.array([[5.0e7, 0.0], [0.0, 1.0e6]])
    y_model = hmm.base.JointObservation({
        'hr_respiration':
            hmm.observe_float.VARG(a, sigma, rng, Psi=Psi, nu=1.0e5),
        'class':
            hmm.base.ClassObservation(class_index2state_indices),
    })

    # Create and return the hmm
    hmm_ = develop.HMM(p_state_initial, p_state_time_average, p_state2state,
                       y_model, args, rng)
    args.read_y_class = hmmds.applications.apnea.model_init.read_lphr_respiration_class
    args.read_raw_y = hmmds.applications.apnea.model_init.read_lphr_respiration

    # Next two lines are for debugging more complicated models
    state_key2state_index = {0: 0, 1: 1}
    state_dict = {0: State([0], [1.0], y_model), 1: State([0], [1.0], y_model)}

    return hmm_, state_dict, state_key2state_index


def read_slow_peak_interval_class(args, record_name):
    """Called by HMM, and returns a dict of observation components

    Args:
        args: From HMM.args
        record_name: EG, 'a01'

    Components are slow, peak, interval and class.

    """
    keys = 'slow peak interval class'.split()
    item_args = {'slow': {'pad': args.AR_order}}

    # develop.HMM.read_y_with_class calls this with self.args, and
    # apnea_train.main wraps the result in hmm.base.JointSegment

    assert args.config.normalize == args.normalize

    hr_instance = utilities.HeartRate(args, record_name, args.config,
                                      args.normalize)
    hr_instance.read_expert()
    hr_instance.filter_hr()
    hr_instance.find_peaks()

    return hr_instance.dict(keys, item_args)


def read_slow_peak_interval(args, record_name):
    """Called by HMM, and returns a dict of observation components

    Args:
        args: From HMM.args
        record_name: EG, 'a01'

    Components are slow, peak, and interval.

    """
    keys = 'slow peak interval'.split()
    item_args = {'slow': {'pad': args.AR_order}}

    # develop.HMM.read_y_with_class calls this with self.args, and
    # apnea_train.main wraps the result in hmm.base.JointSegment

    assert args.config.normalize == args.normalize

    hr_instance = utilities.HeartRate(args, record_name, args.config,
                                      args.normalize)
    hr_instance.filter_hr()
    hr_instance.find_peaks()

    return hr_instance.dict(keys, item_args)


def read_y_class4varg2chain(args, record_name):
    """Called by HMM, and returns a dict of observation components

    Args:
        args: From HMM.args
        record_name: EG, 'a01'

    Components are hr_respiration, peak, interval and class.

    """
    keys = 'hr_respiration peak interval class'.split()

    # develop.HMM.read_y_with_class calls this with self.args, and
    # apnea_train.main wraps the result in hmm.base.JointSegment

    assert args.config.normalize == args.normalize

    hr_instance = utilities.HeartRate(args, record_name, args.config,
                                      args.normalize)
    hr_instance.read_expert()
    hr_instance.filter_hr()
    hr_instance.find_peaks()

    return hr_instance.dict(keys)


def read_raw_y4varg2chain(args, record_name):
    """Called by HMM, and returns a dict of observation components

    Args:
        args: From HMM.args
        record_name: EG, 'a01'

    Components are hr_respiration, peak and interval.

    """
    keys = 'hr_respiration peak interval'.split()

    # develop.HMM.read_y_no_class calls this with self.args, and
    # utilities.ModelRecord wraps the result in hmm.base.JointSegment

    assert args.config.normalize == args.normalize

    hr_instance = utilities.HeartRate(args, record_name, args.config,
                                      args.normalize)
    hr_instance.filter_hr()
    hr_instance.find_peaks()

    return hr_instance.dict(keys)


@register  # low pass heart rate, respiration, and interval
def varg2chain(args, rng):
    """HMM for respiration, heart rate, and interval with one chain
    each for normal and apnea

    """

    #  Normal chain  N Switch A Switch  Apnea chain
    #
    #    **********                     *********
    #   *          \                   /         *
    #  *            \________  _______/           *
    # *peak          |      |--|     |         peak*
    #  *            /--------  -------\           *
    #   *          /                   \         *
    #    **********                     *********

    # ToDo: Think about trainable parameters

    # Functions in utilities normalize heart rate if "'norm_avg' in args"
    args.norm_avg = args.config.norm_avg
    args.read_y_class = hmmds.applications.apnea.model_init.read_y_class4varg2chain
    args.read_raw_y = hmmds.applications.apnea.model_init.read_raw_y4varg2chain

    y_dim = 2
    ar_order = args.AR_order
    coefficient_shape = (y_dim, y_dim * ar_order + 1)
    # Mean of inverse Wishart without data is psi/nu

    coefficients = numpy.zeros(coefficient_shape)
    coefficients[:, 0] = 1.0
    observation_args = {
        'hr_respiration': {
            'coefficients': coefficients,
            'sigma': numpy.eye(2) * 1.0e6,
            'psi': numpy.eye(2) * 1.0e5,
            'nu': 1.0e3
        },
        'class': None,  # Set in copies
        'peak': numpy.array([1.0, 0.0]),
        'interval': None,  # Set in copies
    }

    state_dict = {}

    def make_one_chain(char_class, int_class, pdf_class):
        """
        """
        switch_key = f'{char_class}_switch'
        chain_key = f'{char_class}_chain'
        chain_0 = f'{char_class}_chain_0'

        peak_args = copy.deepcopy(observation_args)
        non_peak_args = copy.deepcopy(observation_args)
        noise_args = copy.deepcopy(observation_args)
        switch_args = copy.deepcopy(observation_args)

        for _args in (peak_args, non_peak_args, noise_args, switch_args):
            _args['class'] = int_class
            _args['interval'] = pdf_class
        peak_args['peak'] = numpy.array([0.0, 1.0])
        noise_args['peak'] = numpy.array([0.5, 0.5])

        peak_chain(switch_key,
                   chain_key,
                   peak_args,
                   non_peak_args,
                   state_dict,
                   length=2)

        make_switch_noise(int_class, [chain_0], state_dict, noise_args,
                          switch_args)

    make_one_chain('N', 0, args.config.normal_pdf)
    make_one_chain('A', 1, args.config.apnea_pdf)

    result_hmm, state_key2state_index = dict2hmm(
        state_dict, make_joint_varg_peak_interval_class, args, rng)

    #for key, value in state_dict.items():
    #    print(f'{key} {value}')

    return result_hmm, state_dict, state_key2state_index


def main(argv=None):
    """Create an hmm and write it as a pickle.
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    rng = numpy.random.default_rng(3)
    with open(args.config, 'rb') as _file:
        args.config = pickle.load(_file)
    args.normalize = args.config.normalize

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
