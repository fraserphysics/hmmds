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


def dict2hmm(state_dict, model_dict, rng, truncate=0):
    """Create an HMM based on state_dict for supervised training

    Args:
        state_dict: state_dict[state_key] is a State instance
        model_dict: Components of joint observation model
        rng: A random number generator
        truncate: Number of elements to drop from the beginning of each segment
                  of class observations.

    Return: hmm, state_key2state_index

    """

    n_states = len(state_dict)
    class_index2state_indices = {}
    p_state_initial = numpy.ones(n_states) / n_states
    p_state_time_average = numpy.ones(n_states) / n_states
    p_state2state = hmm.simple.Prob(numpy.zeros((n_states, n_states)))
    state_key2state_index = {}
    untrainable_indices = []
    untrainable_values = []

    # Build state_key2state_index and class_index2state_indices
    for state_index, (state_key, state) in enumerate(state_dict.items()):
        state_key2state_index[state_key] = state_index
        if state.class_index in class_index2state_indices:
            class_index2state_indices[state.class_index].append(state_index)
        else:
            class_index2state_indices[state.class_index] = [state_index]

    # Build p_state2state
    for state_key, state in state_dict.items():
        state_index = state_key2state_index[state_key]
        for successor_key, probability, trainable in zip(
                state.successors, state.probabilities, state.trainable):
            successor_index = state_key2state_index[successor_key]
            p_state2state[state_index, successor_index] = probability
            if not trainable:
                untrainable_indices.append((state_index, successor_index))
                untrainable_values.append(probability)
    p_state2state.normalize()

    model_dict['class'] = hmm.base.ClassObservation(class_index2state_indices)

    y_model = hmm.base.JointObservation(model_dict, truncate=truncate)

    # Create and return the hmm
    return develop.HMM(p_state_initial,
                       p_state_time_average,
                       p_state2state,
                       y_model,
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


@register  # Models for "a" records
def apnea_dict(args, rng):
    """Return an hmm based on a dict specified in this function.

    The apnea loop has 44 states, a chain of 11 groups of 4.  There is
    a single observation model for all states in each group of 4.  The
    loop starts at "occluded_0" and ends at "last_gasp".

    There is a single "normal" state that has transitions to itself and
    occluded_0.  "last_gasp" has a transition to "occluded_0" and a
    transition to "normal".

    """

    normal_class = 0
    apnea_class = 1

    state_count = 0
    normal = 0
    occluded_0 = 1
    small = 1.0e-10
    state_dict = {
        normal:
            State([normal, occluded_0], [1.0 - 1.0e-3, 1.0e-3], normal_class)
    }
    state_count += 1
    for group in range(11):  # These states are the apnea loop
        if group == 10:
            group_end = occluded_0
        else:
            group_end = state_count + 4
        for _ in range(4):
            state_dict[state_count] = State([state_count + 1, group_end],
                                            [1 - small, small], apnea_class)
            state_count += 1
    last_gasp = state_count - 1
    state_dict[last_gasp] = State([occluded_0, normal], [.99, .01], apnea_class)

    n_states = len(state_dict)

    result, _ = dict2hmm(state_dict,
                         random_observation_model_dict(n_states, args, rng),
                         rng,
                         truncate=args.AR_order)

    result.y_mod.observe([
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_class(args, record))
        for record in args.records
    ])
    result.y_mod.reestimate(
        hmm.simple.Prob(result.y_mod.calculate()).normalize())

    return result


@register  # Alternative models for "a" records
def template_dict(args, rng):
    """An hmm with fixed duration templates and slow states for apnea

    Each template is in a loop with a slow state.  Transitions between
    templates and transitions to the normal state are through the
    "switch" state which is supressed
    
    There is a single "normal" state that has transitions to itself.

    """

    n_normal_states = 3
    n_templates = 6
    template_length = 20

    normal_class = 0
    apnea_class = 1

    # State keys are integers
    switch_state = 0
    first_normal_state = 1
    small = 1.0e-8

    # First normal state connects to switch_state
    state_dict = {}
    transition_probability = (numpy.ones(n_normal_states + 1) -
                              small) / n_normal_states
    transition_probability[0] = small
    state_dict[first_normal_state] = State(numpy.arange(n_normal_states + 1),
                                           transition_probability,
                                           normal_class,
                                           trainable=[False] +
                                           [True] * n_normal_states)
    transitions_from_switch = [first_normal_state]

    # Other normal states all connect with eachother
    normal_state_list = numpy.arange(1, n_normal_states + 1)
    transition_probability = numpy.ones(n_normal_states) / n_normal_states
    for key in range(2, n_normal_states + 1):
        state_dict[key] = State(normal_state_list, transition_probability,
                                normal_class)

    state_dict[switch_state] = None  # Hold place in dict

    state_count = n_normal_states + 1
    for _ in range(n_templates):
        # First state in the template loop
        start_state = state_count
        state_dict[state_count] = State(
            [state_count, state_count + 1, switch_state],
            [1 - 2 * small, 1 - 2 * small, small],
            apnea_class,
            trainable=[True, True, False])
        transitions_from_switch.append(state_count)
        state_count += 1
        # Middle states in the template loop
        for _ in range(1, template_length - 1):
            state_dict[state_count] = State([state_count + 1], [1], apnea_class)
            state_count += 1
        # Last state in the template loop
        state_dict[state_count] = State([start_state], [1], apnea_class)
        state_count += 1

    n_switch = len(transitions_from_switch)
    assert n_switch == n_templates + 1
    state_dict[switch_state] = State(transitions_from_switch,
                                     numpy.ones(n_switch) / n_switch,
                                     apnea_class)
    n_states = len(state_dict)

    result, _ = dict2hmm(state_dict,
                         random_observation_model_dict(n_states, args, rng),
                         rng,
                         truncate=args.AR_order)

    result.y_mod.observe([
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_class(args, record))
        for record in args.records
    ])
    result.y_mod.reestimate(
        hmm.simple.Prob(result.y_mod.calculate()).normalize())

    return result


@register  # Alternative models for "a" records
def fast(args, rng):
    """Return an hmm with multiple fixed duration templates for apnea.

    


    """

    n_normal_states = 3
    template_lengths = numpy.arange(20, 40, 2)

    normal_class = 0
    apnea_class = 1

    # State keys are integers
    apnea_switch_state = n_normal_states
    first_normal_state = 0
    small = 1.0e-8

    # First normal state, state=0, connects to apnea_switch_state
    state_dict = {}
    transition_probability = (numpy.ones(n_normal_states + 1) -
                              small) / n_normal_states
    transition_probability[apnea_switch_state] = small
    state_dict[first_normal_state] = State(numpy.arange(n_normal_states + 1),
                                           transition_probability,
                                           normal_class,
                                           trainable=[True] * n_normal_states +
                                           [False])
    transitions_from_switch = [first_normal_state]

    # Other normal states, [1 : (n_normal_states-1)], all connect with
    # each other
    normal_state_list = numpy.arange(0, n_normal_states)
    transition_probability = numpy.ones(n_normal_states) / n_normal_states
    for key in range(1, n_normal_states):
        state_dict[key] = State(normal_state_list, transition_probability,
                                normal_class)

    state_dict[apnea_switch_state] = None  # Hold place in dict

    # Sets of fast template states.  For each chain,
    # apnea_switch_state connects to the first state and the last
    # state connects back to apnea_switch_state.
    state_count = n_normal_states + 1
    for template_length in template_lengths:
        transitions_from_switch.append(state_count)
        for _ in range(template_length - 1):
            state_dict[state_count] = State([state_count + 1], [1], apnea_class)
            state_count += 1
        state_dict[state_count] = State([apnea_switch_state], [1], apnea_class)
        state_count += 1

    # transitions_from_switch[0] links to the first normal state.
    # Each of the subsequent elements links to the first state of a
    # fast chain.
    n_switch = len(transitions_from_switch)
    p_switch = (numpy.ones(n_switch) - small) / (n_switch - 1)
    p_switch[first_normal_state] = small
    state_dict[apnea_switch_state] = State(transitions_from_switch, p_switch,
                                           apnea_class)
    n_states = len(state_dict)

    result, _ = dict2hmm(state_dict,
                         random_observation_model_dict(n_states, args, rng),
                         rng,
                         truncate=args.AR_order)

    result.y_mod.observe([
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_class(args, record))
        for record in args.records
    ])
    result.y_mod.reestimate(
        hmm.simple.Prob(result.y_mod.calculate()).normalize())

    return result


def make_chains(chain_lengths, switch_key: str, other_key: str, int_class: int,
                state_dict):
    """Add a sequence of states to state_dict

    Args:
        chain_lengths:  Length of each fast sequence
        switch_key: Key of state that links these chains
        other_key: Key of state that links other class
        int_class:
        state_dict:

    """

    if int_class == 0:
        letter_class = 'N'
    else:
        letter_class = 'A'
    switch_transitions = [other_key]
    for chain_length in chain_lengths:
        state_key = f'{letter_class}_{chain_length}_0'
        switch_transitions.append(state_key)
        for i in range(1, chain_length):
            next_state_key = f'{letter_class}_{chain_length}_{i}'
            state_dict[state_key] = State([next_state_key], [1], int_class)
            state_key = next_state_key
        state_dict[state_key] = State([switch_key], [1], int_class)
    state_dict[switch_key] = State(
        switch_transitions,
        numpy.ones(len(switch_transitions)) / len(switch_transitions),
        int_class)


@register  # Alternative models for "a" records
def balanced(args, rng):
    """Return an hmm with multiple fixed duration chains for both
    normal and apnea.

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

    chain_lengths = numpy.arange(20, 40, 2)
    n_block = chain_lengths.sum()

    normal_class = 0
    apnea_class = 1

    state_dict = {}

    make_chains(chain_lengths, 'normal_switch', 'apnea_switch', normal_class,
                state_dict)

    make_chains(chain_lengths, 'apnea_switch', 'normal_switch', apnea_class,
                state_dict)

    result_hmm, state_key2state_index = dict2hmm(state_dict,
                                                 random_observation_model_dict(
                                                     len(state_dict), args,
                                                     rng),
                                                 rng,
                                                 truncate=args.AR_order)

    result_hmm.y_mod.observe([
        hmm.base.JointSegment(
            hmmds.applications.apnea.utilities.read_slow_class(args, record))
        for record in args.records
    ])
    result_hmm.y_mod.reestimate(
        hmm.simple.Prob(result_hmm.y_mod.calculate()).normalize())

    return result_hmm, state_dict, state_key2state_index


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
