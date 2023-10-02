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
import develop
import hmmds.applications.apnea.utilities
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


def make_joint_slow_peak_interval_class(state_dict, keys, rng, truncate=0):
    """Return a JointObservation instance with components "slow",
    "peak", "interval", and "class"

    Args:
        state_dict:
        keys: Establishes order for state_dict
        rng:
        truncate:

    Return: a JointObservation instance

    result["slow"] is an AutoRecressive instance
    result["peak"] is an IntegerObservation instance
    result['interval'] is ???
    result['class'] is ???

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

    return hmm.base.JointObservation(
        {
            'slow':
                hmm.C.AutoRegressive(ar_coefficients, offsets, variances, rng,
                                     alphas, betas),
            'peak':
                hmm.C.IntegerObservation(py_state, rng),
            'interval':
                hmmds.applications.apnea.utilities.IntervalObservation(
                    tuple(
                        state_dict[key].observation['interval'] for key in keys)
                ),
            'class':
                hmm.base.ClassObservation(class_index2state_indices),
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


def peak_chain(switch_key: str,
               prefix: str,
               int_class: int,
               pdf_interval: callable,
               args,
               rng,
               state_dict,
               peak_prob: numpy.ndarray,
               length=9):
    """Add a sequence of states to state_dict

    Args:
        switch_key: Key of state that links these chains
        prefix: For state keys
        int_class:
        args:
        rng:
        state_dict:
        peak_prob: Output distribution at peak
        length: Number of 2.5 second samples in peak pattern

    """

    # Magic numbers

    args_alpha, args_beta = args.alpha_beta

    variance = 1.0e3
    non_peak = numpy.zeros(len(peak_prob))
    non_peak[0] = 1.0

    def make_observation(alpha, beta, prob):
        """Return an argument for State.__init__()
        """
        return {
            'slow': {
                'coefficients': rng.random(args.AR_order) / args.AR_order,
                'alpha': alpha,
                'beta': beta,
                'offset': 0.0,
                'variance': variance
            },
            'peak': prob,
            'interval': pdf_interval,
            'class': int_class,
        }

    for index in range(length + 1):
        state_key = f'{prefix}_{index}'
        next_key = f'{prefix}_{index+1}'
        if index == int(length / 2):
            prob = peak_prob
        else:
            prob = non_peak
        state_dict[state_key] = State([next_key], [1.0],
                                      make_observation(args_alpha, args_beta,
                                                       prob))

    # Repair transitions in slow and last states
    slow_key = f'{prefix}_0'
    first_key = f'{prefix}_1'
    last_key = f'{prefix}_{length}'
    state_dict[slow_key].set_transitions([switch_key, slow_key, first_key],
                                         [1.0e-20, .8, .2],
                                         trainable=(False, True, True))
    state_dict[last_key].set_transitions([slow_key], [1.0])


def make_switch_noise(args, int_class, chain_keys, state_dict,
                      pdf_interval: callable, peak_dimension, rng):
    """Make states for switching and for noise

    Args:
        int_class: Either 0 or 1
        chain_keys: Names of states that switch links to
        state_dict:
    """
    # Magic numbers
    noise_p = 1.0e-30  # Probability of transition to noise state
    # from self or either switch state
    switch_p = 1.0e-10  # Probability of transition between classes
    p_switch_self = 1.0e-30  # Probability of tranistion from
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

    # Define alpha and beta of inverse gamma for noise states.
    # There are about 12,000 minutes of data, 25 records * 480 minutes
    noise_alpha = 9.6e4
    noise_prior_variance = 10.0**2
    noise_beta = noise_alpha * noise_prior_variance

    args_alpha, args_beta = args.alpha_beta

    variance = 1.0e3

    def make_observation(alpha: float, beta: float, p_y: numpy.ndarray) -> dict:
        """Makes dict for switch and noise states"""
        return {
            'slow': {
                'coefficients': rng.random(args.AR_order) / args.AR_order,
                'alpha': alpha,
                'beta': beta,
                'offset': 0.0,
                'variance': variance
            },
            'peak': p_y,
            'interval': pdf_interval,
            'class': int_class,
        }

    p_y = numpy.ones(peak_dimension) / peak_dimension

    # Specify noise state
    state_dict[noise_key] = State([noise_key, other_noise, switch_key],
                                  [noise_p, noise_p, 1.0 - 2 * noise_p],
                                  make_observation(noise_alpha, noise_beta,
                                                   p_y),
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
    p_y = numpy.zeros(peak_dimension)
    p_y[0] = 1
    state_dict[switch_key] = State(successors,
                                   probabilities,
                                   make_observation(args_alpha, args_beta, p_y),
                                   trainable=trainable)


@register  # Joint observation includes values for peaks
def hmm_intervals(args, rng):
    """Return an hmm for joint observations that include "peaks"

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

    peak_dict, boundaries = hmmds.applications.apnea.utilities.peaks_intervals(
        args, args.a_names)
    # Attach information to args for creating observation models and
    # reading observations
    args.boundaries = boundaries
    args.read_y_class = hmmds.applications.apnea.utilities.read_slow_class_peak_interval
    args.read_raw_y = hmmds.applications.apnea.utilities.read_slow_peak_interval
    peak_dimension = len(boundaries) + 1  # Dimension of output for peaks

    normal_class = 0
    apnea_class = 1

    state_dict = {}
    interval_pdfs = hmmds.applications.apnea.utilities.make_interval_pdfs(args)
    # Make the one chain for modeling normal peaks
    normal_peak_prob = numpy.ones(peak_dimension) / (peak_dimension - 1)
    normal_peak_prob[0] = 0
    peak_chain('N_switch', 'N_chain', normal_class, interval_pdfs.normal_pdf,
               args, rng, state_dict, normal_peak_prob)

    make_switch_noise(args, normal_class, ['N_chain_0'], state_dict,
                      interval_pdfs.normal_pdf, peak_dimension, rng)

    # Set up discrete p_y for apnea peaks
    peak_probs = numpy.zeros((len(boundaries), len(boundaries) + 1))
    peak_probs[:, 1:] = numpy.eye(len(boundaries))

    # Make chains for apnea peaks
    a_chain_links = []
    for number, peak_prob in enumerate(peak_probs):
        chain_key = f'A{number}'
        a_chain_links.append(f'{chain_key}_0')
        peak_chain('A_switch', chain_key, apnea_class, interval_pdfs.apnea_pdf,
                   args, rng, state_dict, peak_prob)

    make_switch_noise(args, apnea_class, a_chain_links, state_dict,
                      interval_pdfs.apnea_pdf, peak_dimension, rng)

    result_hmm, state_key2state_index = dict2hmm(
        state_dict,
        make_joint_slow_peak_interval_class,
        rng,
        truncate=args.AR_order)
    return result_hmm, state_dict, state_key2state_index


@register  # For debugging effect of intervals on classification
def two_intervals(args, rng):
    """Return an hmm for joint observations that include "peaks"

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

    peak_dict, boundaries = hmmds.applications.apnea.utilities.peaks_intervals(
        args, args.a_names)
    # Attach information to args for creating observation models and
    # reading observations
    args.boundaries = boundaries
    args.read_y_class = hmmds.applications.apnea.utilities.read_slow_class_peak_interval
    args.read_raw_y = hmmds.applications.apnea.utilities.read_slow_peak_interval
    peak_dimension = len(boundaries) + 1  # Dimension of output for peaks

    normal_class = 0
    apnea_class = 1

    state_dict = {}
    interval_pdfs = hmmds.applications.apnea.utilities.make_interval_pdfs(args)

    def make_one_chain(char_class, int_class, pdf_class):
        """
        """
        switch_key = f'{char_class}_switch'
        chain_key = f'{char_class}_chain'
        chain_0 = f'{char_class}_chain_0'
        peak_prob = numpy.ones(peak_dimension) / (peak_dimension - 1)
        peak_prob[0] = 0
        peak_chain(switch_key, chain_key, int_class, pdf_class, args, rng,
                   state_dict, peak_prob)

        make_switch_noise(args, int_class, [chain_0], state_dict, pdf_class,
                          peak_dimension, rng)

    make_one_chain('N', 0, interval_pdfs.normal_pdf)
    make_one_chain('A', 1, interval_pdfs.apnea_pdf)
    #for key, value in state_dict.items():
    #    print(f'{key} {value}')

    result_hmm, state_key2state_index = dict2hmm(
        state_dict,
        make_joint_slow_peak_interval_class,
        rng,
        truncate=args.AR_order)
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
    assert model.p_state_initial.min() > 0

    model.strip()
    args.state_dict = state_dict
    args.state_key2state_index = state_key2state_index
    with open(args.write_path, 'wb') as _file:
        pickle.dump((args, model), _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
