"""model_init.py Create initial HMM models with apnea observations

A rule modified from Rules.mk:

${MODELS}/initial_%: model_init.py utilities.py
	python model_init.py --root ${ROOT} $* $@

The pattern % or $* selects one of the registered functions in this
module.

"""
from __future__ import annotations  # Enables, eg, (self: HMM,

import sys
import os.path
import pickle
import argparse

import numpy

import hmm.base
import hmm.simple

import hmm.C
import hmmds.applications.apnea.ECG.utilities


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Create and write/pickle an initial model")
    hmmds.applications.apnea.ECG.utilities.common_arguments(parser)
    # args.records is None if --records is not on command line
    parser.add_argument(
        "--ecg_alpha_beta",
        type=float,
        nargs=2,
        default=(1.0e3, 1.0e2),
        help=
        "Paramters of inverse gamma prior for variance for normal ecg signal")
    parser.add_argument("--noise_parameters",
                        type=float,
                        nargs=3,
                        default=(1.0e8, 1.0e10, 1.0e-10),
                        help="Outlier model: alpha, beta, noise probability")
    parser.add_argument('--tag_ecg',
                        action='store_true',
                        help="Invoke tagging in utilities.read_ecgs()")
    parser.add_argument('--AR_order',
                        type=int,
                        default=3,
                        help="Number of previous values for prediction.")
    parser.add_argument(
        '--before_after_slow',
        nargs=3,
        type=int,
        default=(18, 30, 3),
        help=
        "Number of transient states before and after R in ECG, and number of slow states."
    )

    parser.add_argument('--records',
                        type=str,
                        nargs='+',
                        help='--records a01 x02 -- ')
    parser.add_argument(
        'key',
        type=str,
        help='One of the functions registered in the source, eg, A4')
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    hmmds.applications.apnea.ECG.utilities.join_common(args)
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


class HMM(hmm.C.HMM):
    """Holds state transition probabilities constant

    """

    def __init__(self: HMM,
                 *args,
                 untrainable_indices=None,
                 untrainable_values=None):
        """Option of holding some elements of p_state2state constant
        in reestimation.

        """
        hmm.C.HMM.__init__(self, *args)
        self.untrainable_indices = untrainable_indices
        self.untrainable_values = untrainable_values

    def reestimate(self: HMM):
        """Variant can hold some self.p_state2state values constant.

        Reestimates observation model parameters.

        """

        hmm.C.HMM.reestimate(self)
        if self.untrainable_indices is None:
            return
        self.p_state2state[self.untrainable_indices] = self.untrainable_values
        return

    def likelihood(self: HMM, y) -> numpy.ndarray:
        """Calculate p(y[t]|y[:t]) for t < len(y)

        Args:
            y: A single segment appropriate for self.y_mod.observe([y])

        """
        self.y_mod.observe([y])
        state_likelihood = self.y_mod.calculate()
        length = len(state_likelihood)  # Less than len(y) if y_mod is
        # autoregressive
        result = numpy.empty(length)
        last = numpy.copy(self.p_state_initial)
        for t in range(length):
            last *= state_likelihood[t]
            last_sum = last.sum()  # Probability of y[t]|y[:t]
            result[t] = last_sum
            if last_sum > 0.0:
                last /= last_sum
            else:
                print(f'Zero likelihood at {t=}.  Reset.')
                last = numpy.copy(self.p_state_initial)
            self.p_state2state.step_forward(last)
        return result


def _make_hmm(y_model,
              p_state_initial,
              p_state_time_average,
              p_state2state,
              names,
              args,
              rng,
              Class=HMM):
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

    y_data = hmmds.applications.apnea.ECG.utilities.list_heart_rate_respiration_data(
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
        self.trainable = trainable
        # Each class_index must be an int because the model will be a
        # subclass of hmm.base.IntegerObservation


def dict2hmm(state_dict, ecg_model, rng):
    """Create an HMM based on state_dict for supervised training

    Args:
        state_dict: state_dict[state_name] is a State instance
        ecg_model: Observation model for raw ecg samples
        rng: A random number generator

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

    if "bad" in state_dict:
        n_classes = len(class_index2state_indices)
        likelihood = 1.0 / n_classes
        # Given the bad state all classes have the same likelihood
        bad2class = {
            state_name2state_index["bad"]:
                list((class_index, likelihood)
                     for class_index in range(n_classes))
        }
        class_model = hmm.base.BadObservation(class_index2state_indices,
                                              bad2class)
    else:
        class_model = hmm.base.ClassObservation(class_index2state_indices)

    y_model = hmm.base.JointObservation({
        "class": class_model,
        "ecg": ecg_model
    })

    # Create and return the hmm
    indices = tuple(numpy.array(untrainable_indices).T)
    return HMM(p_state_initial,
               p_state_time_average,
               p_state2state,
               y_model,
               rng,
               untrainable_indices=indices,
               untrainable_values=numpy.array(
                   untrainable_values)), state_name2state_index


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


@register
def masked_dict(args, rng):
    """Return an hmm based on a dict specified in this function.

    """

    alpha, beta = args.ecg_alpha_beta
    noise_alpha, noise_beta, p_noise = args.noise_parameters
    n_before, n_after, n_slow = args.before_after_slow
    slow_class = 0
    first_fast = -n_before
    last_slow = f'slow_{n_slow-1}'
    bad = "bad"

    # The bad state is for outliers
    bad_state = State([bad, 'slow_0'], [1.0 - p_noise, p_noise], slow_class,
                      [False, False])
    state_dict = {bad: bad_state}  # Maps state name to State instance

    # Define slow states with transitions to themselves.  These states
    # model the variable intervals between PQRST sequences
    for i in range(n_slow - 1):
        state_dict[f'slow_{i}'] = State(
            [f'slow_{i}', f'slow_{i+1}', first_fast, bad],
            [.4 - p_noise / 2, .4 - p_noise / 2, .2, p_noise], slow_class,
            [True, True, True, False])
    state_dict[last_slow] = State([last_slow, first_fast, bad],
                                  [.5 - p_noise, .5, p_noise], slow_class,
                                  [True, True, False])

    # Define fast states.  This is fit to the PQRST sequence.  In a
    # normal ECG, each instance of the PQRST sequence has about the
    # same duration.  In this hmm the sequence of fast states normally
    # progresses through one state each time step.
    for t in range(first_fast, n_after):
        state_dict[t] = State(
            [t + 1, bad],  # successors
            [1.0 - p_noise, p_noise],  # probabilities
            t + n_before + 1,  # class
            [False, False]  # Trainable
        )

    # Close the loop by connecting the last fast state to the first slow state
    state_dict[n_after] = State(['slow_0', bad], [1.0 - p_noise, p_noise],
                                n_after + n_before + 1, [False, False])

    class_set = set((state.class_index for state in state_dict.values()))
    if class_set != set(range(len(class_set))):
        raise RuntimeError(
            f"Classes are not sequential integers.  {class_set=}")

    n_states = len(state_dict)
    ar_coefficients = numpy.ones((n_states, args.AR_order)) / args.AR_order
    offset = numpy.zeros(n_states)
    variances = numpy.ones(n_states)
    # I think right variance is between .05 and .001, and there are
    # about 50,000 samples per state in a01

    ecg_model = hmm.C.AutoRegressive(ar_coefficients,
                                     offset,
                                     variances,
                                     rng,
                                     alpha=numpy.ones(n_states) * alpha,
                                     beta=numpy.ones(n_states) * beta)

    result, state_name2state_index = dict2hmm(state_dict, ecg_model, rng)
    # Force the variance of the bad state to be special
    i_bad = state_name2state_index['bad']
    ecg_model.alpha[i_bad] = noise_alpha
    ecg_model.beta[i_bad] = noise_beta

    # Initialize the y_model parameters based on the data
    y_data = hmmds.applications.apnea.ECG.utilities.read_ecgs(args)
    result.y_mod.observe(y_data)
    weights = hmm.simple.Prob(result.y_mod.calculate()).normalize()
    result.y_mod.reestimate(weights)

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
