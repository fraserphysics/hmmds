"""model_init.py Create initial HMM models with apnea observations

A rule from Rules.mk:

$(MODELS)/a%_initial: model_init.py
	python $< --records a$* --alpha_beta 1.0e3 1.0e3 --trim_start 25 apnea_dict $@


"""
import sys
import os.path
import pickle
import argparse

import numpy

import hmm.base
import hmm.simple

import hmm.C
import hmmds.applications.apnea.utilities
import develop
from hmmds.applications.apnea.utilities import State


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Create and write/pickle an initial model")
    hmmds.applications.apnea.utilities.common_arguments(parser)
    parser.add_argument("--alpha_beta",
                        type=float,
                        nargs=2,
                        default=(1.0e1, 2.0e1),
                        help="Paramters of inverse gamma prior for variance")
    parser.add_argument('--AR_order',
                        type=int,
                        default=1,
                        help="Number of previous values for prediction.")
    parser.add_argument('--boundaries',
                        type=str,
                        default='boundaries',
                        help="Path to levels for key peaks")

    parser.add_argument(
        'key',
        type=str,
        help='One of the functions registered in the source, eg, apnea_dict')
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.utilities.join_common(args)
    return args


def random_1d_prob(rng: numpy.random.Generator, length: int) -> numpy.ndarray:
    """Draw a probability vector from a uniform distribution
    Args:
       rng: Random number generator
       length: Dimension of returned vector

    Returns:
        A normalized probability mass function
    """
    vector = rng.random(length)
    return vector / vector.sum()


def random_conditional_prob(rng: numpy.random.Generator,
                            shape: tuple) -> hmm.simple.Prob:
    """Draw a random conditional distribution P_{a|b}
    Args:
       rng: Random number generator
       shape: (|A|,|B|) where |A| is the number of possible values of a

    Returns:
        A normalized conditional distribution
    """
    return hmm.simple.Prob(rng.random(shape)).normalize()


def _make_hmm(y_model,
              p_state_initial,
              p_state_time_average,
              p_state2state,
              names,
              args,
              rng,
              Class=develop.hmm):
    """Create a hmm using parameters defined in the caller

    Args:
        y_model: P(y[t]|s[t]) and functions to support reestimation
        p_state_initial:
        p_state_time_average:
        p_state2state: p[t=1] = numpy.dot(p[t=0], p_state2state)
        names: List of record names, eg, 'c01 a05 x35'.split()
        rng: Random number generator

    Return: hmm initialized with data specified by names

    Unsupervised, ie, no class information. AR-4 for heart rate. 3-d
    multivariate Gaussian for respiration.

    """

    _hmm = Class.HMM(p_state_initial, p_state_time_average, p_state2state,
                     y_model, rng)

    y_data = hmmds.applications.apnea.utilities.list_heart_rate_respiration_data(
        names, args)
    _hmm.initialize_y_model(y_data)
    return _hmm


def make_joint_peak_class_slow(state_dict):
    """Return a JointObservation instance with components "peak",
    "class" and "slow"

    """
    pass


def make_joint_class_slow(state_dict, keys, rng, truncate=0):
    """Return a JointObservation instance with components "class" and
    "slow"

    Args:
        state_dict:
        keys: Establishes order for state_dict
        rng:

    """
    class_index2state_indices = {}
    n_states = len(keys)
    assert n_states == len(state_dict)
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

    slow_model = hmm.C.AutoRegressive(ar_coefficients, offsets, variances, rng,
                                      alphas, betas)
    return hmm.base.JointObservation(
        {
            'slow': slow_model,
            'class': hmm.base.ClassObservation(class_index2state_indices)
        },
        truncate=truncate)


def dict2hmm(state_dict, make_observation_model, rng, truncate=0):
    """Create an HMM based on state_dict for supervised training

    Args:
        state_dict: state_dict[state_key] is a State instance
        make_observation_model: Function
        rng: A random number generator
        truncate: Number of elements to drop from the beginning of each segment
                  of class observations.

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
                       make_observation_model(state_dict,
                                              state_keys,
                                              rng,
                                              truncate=truncate),
                       rng,
                       untrainable_indices=tuple(
                           numpy.array(untrainable_indices).T),
                       untrainable_values=numpy.array(
                           untrainable_values)), state_key2state_index


def random_observation_model_dict(n_states, args, rng):
    """Return model for observations with random coefficients

    """
    # Number of data points for each state is going to be about 500
    ar_coefficients = rng.random((n_states, args.AR_order)) / args.AR_order
    offset = numpy.zeros(n_states)
    variances = numpy.ones(n_states) * 1e3
    slow_model = hmm.C.AutoRegressive(
        ar_coefficients.copy(),
        offset.copy(),
        variances.copy(),
        rng,
        alpha=numpy.ones(n_states) * args.alpha_beta[0],
        beta=numpy.ones(n_states) * args.alpha_beta[1])

    return {'slow': slow_model}


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


@register  # Models for "c" records
def c_model(args, rng):
    """Return an hmm based on a dict specified in this function.

    This creates a model with a structure like a model from apnea_dict
    so that the likelihoods of the models will be easy to compare.

    """

    n_states = 4
    p_state_initial = numpy.ones(n_states)
    p_state_time_average = numpy.ones(n_states) / n_states
    p_state2state = hmm.simple.Prob(numpy.ones(
        (n_states, n_states))) + numpy.eye(n_states) + numpy.roll(
            numpy.eye(4), 1, axis=1)
    # break_symmetry
    p_state_initial[0] = 10
    p_state_initial /= p_state_initial.sum()
    p_state2state.normalize()

    y_model = hmm.base.JointObservation(random_observation_model_dict(
        n_states, args, rng),
                                        truncate=args.AR_order)

    # Create and return the hmm
    return develop.HMM(p_state_initial, p_state_time_average, p_state2state,
                       y_model, rng)


# Need state_dict and observation_dict
class Peak:
    """A slow state and a deterministic sequence of states

    Args:
        switch_key
        length

    """

    def __init__(self, switch_key, length):
        pass


def make_chains(lengths_names, switch_key: str, other_key: str, int_class: int,
                args, rng, state_dict):
    """Add a sequence of states to state_dict

    Args:
        lengths_names:  Length and name of each fast sequence
        switch_key: Key of state that links these chains
        other_key: Key of state that links other class
        int_class:
        args:
        rng:
        state_dict:

    """

    # Magic numbers
    noise_p = 1.0e-30  # Probability of transition to noise state
    switch_p = 1.0e-6  # Probability of transition between classes

    # Define alpha and beta of inverse gamma for noise states.
    # There are about 12,000 minutes of data, 25 records * 480 minutes
    noise_alpha = 9.6e4
    noise_prior_variance = 10.0**2
    noise_beta = noise_alpha * noise_prior_variance

    args_alpha, args_beta = args.alpha_beta

    variance = 1.0e3
    if int_class == 0:
        letter_class = 'N'
    else:
        letter_class = 'A'
    noise_key = f'{letter_class}_noise'

    def make_observation(alpha, beta):
        """Return an argument for State.__init__()
        """
        return {
            'class': int_class,
            'slow': {
                'coefficients': rng.random(args.AR_order) / args.AR_order,
                'alpha': alpha,
                'beta': beta,
                'offset': 0.0,
                'variance': variance
            }
        }

    state_dict[noise_key] = State([noise_key, switch_key],
                                  [noise_p, 1.0 - noise_p],
                                  make_observation(noise_alpha, noise_beta),
                                  trainable=(False, False))

    switch_transitions = [other_key, noise_key]
    for (length, name) in lengths_names:
        slow_key = f'{letter_class}_{name}_0'
        state_key = f'{letter_class}_{name}_1'
        state_dict[slow_key] = State([switch_key, slow_key, state_key],
                                     [.1, .8, .1],
                                     make_observation(args_alpha, args_beta))
        switch_transitions.append(slow_key)

        # Create states in the chain
        for i in range(2, length):
            next_state_key = f'{letter_class}_{name}_{i}'
            state_dict[state_key] = State([next_state_key], [1],
                                          make_observation(
                                              args_alpha, args_beta))
            state_key = next_state_key
        state_dict[state_key] = State([slow_key], [1],
                                      make_observation(args_alpha, args_beta))
    p_switch = numpy.ones(
        len(switch_transitions)) / (len(switch_transitions) - 2)
    p_switch[0] = switch_p
    p_switch[1] = noise_p
    trainable = [True] * len(switch_transitions)
    trainable[0:2] = (False, False)
    state_dict[switch_key] = State(switch_transitions,
                                   p_switch,
                                   make_observation(args_alpha, args_beta),
                                   trainable=trainable)


@register  # Alternative models for "a" records
def balanced(args, rng):
    """Return an hmm with multiple fixed duration chains for both
    normal and apnea.

    2023-07-21: Make chains short to model positive pulse rate peaks.
    Here are the durations of peaks that I saw some records:

    Record   Duration     Samples
             in minutes   at 24 cpm

    a01      .3
    a02      .3
    a03      .35          8.40
    a04      .22          5.28
    a05      .3

    After looking at plots of the filtered heart rate, I chose
    arguments --heart_rate_sample_frequency 24 --low_pass_period 8

    """

    #  Normal Chains  N Switch A Switch  Apnea Chains
    #
    # *************                     ************
    #              \                   /
    #               \________  _______/
    #                |      |--|     |
    #               /--------  -------\
    #              /                   \
    # *************                     ************

    normal_chains = (
        (1, '1'),
        (6, '6'),
    )
    apnea_chains = (
        (6, '6'),
        (7, '7'),
    )

    normal_class = 0
    apnea_class = 1

    state_dict = {}

    make_chains(normal_chains, 'N_switch', 'A_switch', normal_class, args, rng,
                state_dict)

    make_chains(apnea_chains, 'A_switch', 'N_switch', apnea_class, args, rng,
                state_dict)

    observation_model = random_observation_model_dict(len(state_dict), args,
                                                      rng)

    result_hmm, state_key2state_index = dict2hmm(state_dict,
                                                 make_joint_class_slow,
                                                 rng,
                                                 truncate=args.AR_order)

    for key, index in state_key2state_index.items():
        if state_dict[key].prior is None:
            continue
        observation_model['slow'].alpha[index] = state_dict[key].prior[0]
        observation_model['slow'].beta[index] = state_dict[key].prior[1]

    result_hmm.y_mod.observe([
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_class(args, record))
        for record in args.records
    ])
    result_hmm.y_mod.reestimate(
        hmm.simple.Prob(result_hmm.y_mod.calculate()).normalize())

    return result_hmm, state_dict, state_key2state_index


@register  # Joint observation includes values for peaks
def peaks(args, rng):
    """Return an hmm with multiple single state nodes for both
    normal and apnea.

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

    with open(args.boundaries, 'rb') as _file:
        boundaries = pickle.load(_file)
    print(f'{boundaries=}')
    sys.exit(0)
    #return result_hmm, state_dict, state_key2state_index


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    rng = numpy.random.default_rng(3)

    # Run the function specified by args.key
    model, state_dict, state_key2state_index = MODELS[args.key](args, rng)
    assert model.p_state_initial.min() > 0

    model.strip()
    args.state_dict = state_dict
    args.state_key2state_index = state_key2state_index
    with open(args.write_path, 'wb') as _file:
        pickle.dump((args, model), _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
