"""model_init.py Create initial HMM models with apnea observations

A rule modified from Rules.mk:

${MODELS}/initial_%: model_init.py utilities.py observation.py
	python model_init.py --root ${ROOT} $* $@

The pattern % or $* selects one of the registered functions in this
module.

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
import hmmds.applications.apnea.observation
import develop
import utilities


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


class State:
    """For defining HMM graph

    Args:
        successors: List of names (dict keys) of successor states
        probabilities: List of float probabilities for successors
        class_index: Integer class
        trainable: List of True/False for transitions described above
    """

    def __init__(self, successors, probabilities, class_index, trainable=None):
        self.successors = successors
        self.probabilities = probabilities
        self.class_index = class_index
        if trainable:
            self.trainable = trainable
        else:
            self.trainable = [True] * len(successors)
        # Each class_index must be an int because the model will be a
        # subclass of hmm.base.IntegerObservation


def dict2hmm(state_dict, model_dict, rng, truncate=0):
    """Create an HMM based on state_dict for supervised training

    Args:
        state_dict: state_dict[state_name] is a State instance
        model_dict: Components of joint observation model
        rng: A random number generator
        truncate: Number of elements to drop from the beginning of each segment
                  of class observations.

    """

    n_states = len(state_dict)
    class_index2state_indices = {}
    p_state_initial = numpy.ones(n_states) / n_states
    p_state_time_average = numpy.ones(n_states) / n_states
    p_state2state = hmm.simple.Prob(numpy.zeros((n_states, n_states)))
    state_name2state_index = {}
    untrainable_indices = []
    untrainable_values = []

    # Build state_name2state_index and class_index2state_indices
    for state_index, (state_name, state) in enumerate(state_dict.items()):
        state_name2state_index[state_name] = state_index
        if state.class_index in class_index2state_indices:
            class_index2state_indices[state.class_index].append(state_index)
        else:
            class_index2state_indices[state.class_index] = [state_index]

    # Build p_state2state
    for state_name, state in state_dict.items():
        state_index = state_name2state_index[state_name]
        for successor_name, probability, trainable in zip(
                state.successors, state.probabilities, state.trainable):
            successor_index = state_name2state_index[successor_name]
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
                           untrainable_values)), state_name2state_index


def initial_model_dict(n_states, args, rng):
    """Return model for observations

    """
    ar_coefficients = numpy.ones((n_states, args.AR_order)) / args.AR_order
    offset = numpy.zeros(n_states)
    variances = numpy.ones(n_states) * 1e3
    slow_model = hmm.C.AutoRegressive(
        ar_coefficients.copy(),
        offset.copy(),
        variances.copy(),
        rng,
        alpha=numpy.ones(n_states) * args.alpha_beta[0],
        beta=numpy.ones(n_states) * args.alpha_beta[1])
    respiration_model = hmm.C.AutoRegressive(
        ar_coefficients.copy(),
        offset.copy(),
        variances.copy(),
        rng,
        alpha=numpy.ones(n_states) * args.alpha_beta[0],
        beta=numpy.ones(n_states) * args.alpha_beta[1])

    return {'slow': slow_model, 'respiration': respiration_model}


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

    y_model = hmm.base.JointObservation(initial_model_dict(n_states, args, rng),
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

    # Number of data points for each state is going to be about 500
    ar_coefficients = numpy.ones((n_states, args.AR_order)) / args.AR_order
    offset = numpy.zeros(n_states)
    variances = numpy.ones(n_states) * 1e3

    result, _ = dict2hmm(state_dict,
                         initial_model_dict(n_states, args, rng),
                         rng,
                         truncate=args.AR_order)

    # ToDo: Create observation models with these characteristics:

    # Observation models are joint slow, respiration and class.  Slow models
    # the heart rate oscillations that match the occlusion - gasp
    # cycle, and respiration models catch the ~14 cycle per minute
    # respiration signal.

    # There is a single observation model for each group of 4 states
    # in the apnea loop.

    # Initialize the y_model parameters based on the data sampled with
    # a period of 1.5 seconds or 40 samples per minute.

    result.y_mod.observe([
        hmm.base.JointSegment(
            utilities.read_slow_respiration_class(args, record))
        for record in args.records
    ])
    result.y_mod.reestimate(
        hmm.simple.Prob(result.y_mod.calculate()).normalize())

    return result


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    rng = numpy.random.default_rng(3)

    # Run the function specified by args.key
    model = MODELS[args.key](args, rng)
    assert model.p_state_initial.min() > 0

    model.strip()
    with open(args.write_path, 'wb') as _file:
        pickle.dump((args, model), _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
