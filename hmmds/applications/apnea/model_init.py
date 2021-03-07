"""model_init.py Create initial HMM models with apnea observations

From Makefile:

python model_init.py $* $@

$* is one of: A2 C1 High Medium Low
"""
import sys
import os.path
import pickle
import argparse

import numpy

import hmm.base

import hmmds.applications.apnea.utilities
import observation
import develop


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


# State transition probabilities for the Low and Medium models
P_SS_LowMedium = hmm.simple.Prob(
    numpy.array(
        [
            #1  2  3  4  5  6  7  8  9  10
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #1
            [1, 1, 1, 1, 1, 0, 1, 0, 0, 0],  #2
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  #3
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],  #4
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  #5
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  #6
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],  #7
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  #8
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],  #9
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  #10
        ],
        dtype=numpy.float64))

# State transition probabilities for the High model
P_SS_High = hmm.simple.Prob(
    numpy.array(
        [
            #1  2  3  4  5  6  7  8  9  10 11 12 13 14
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1
            [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],  #2
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #3
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #4
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #5
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #6
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #7
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  #8
            [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],  #9
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  #10
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],  #11
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],  #12
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  #13
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  #14
        ],
        dtype=numpy.float64),)


def filtered_heart_rate_model(n_states, rng, ar_order=4):
    """Make an initial observation model
    """
    offset = numpy.zeros(n_states)
    variance = numpy.ones(n_states)
    ar_coefficients = numpy.ones((n_states, ar_order)) / ar_order
    return observation.FilteredHeartRate(ar_coefficients, offset, variance, rng)


def respiration_model(n_states, rng, dimension=3):
    """Make an initial observation model
    """
    mu = numpy.zeros((n_states, dimension))
    sigma = numpy.zeros((n_states, dimension, dimension))
    for state in range(n_states):
        sigma[state, :, :] = numpy.eye(dimension)
    return observation.Respiration(mu, sigma, rng)


def filtered_heart_rate_respiration_bundle_model(n_states,
                                                 bundle2state,
                                                 rng,
                                                 ar_order=4,
                                                 dimension=3):
    """Make an initial observation model
    """
    hr_mod = filtered_heart_rate_model(n_states, rng, ar_order=ar_order)
    res_mod = respiration_model(n_states, rng, dimension=dimension)

    underlying_model = observation.FilteredHeartRate_Respiration(
        hr_mod, res_mod, rng)

    return hmm.base.Observation_with_bundles(underlying_model, bundle2state,
                                             rng)


MODELS = {}  # Is populated by @register decorated functions.  The keys
# are function names, and the values are functions


def register(func):
    """Decorator that puts function in MODELS dictionary"""
    #See https://realpython.com/primer-on-python-decorators/
    MODELS[func.__name__] = func
    return func


@register
def A2(common, rng) -> develop.HMM:
    """Two states, no bundles, AR-4 for heart rate, single Gaussian for
    respiration

    Args:
        rng: A numpy random number generator instance
        common: A Collection of information for the apnea project.

    Use data from a01 to initialize parameters of the observation model
    """
    n_states = 2

    y_model = observation.FilteredHeartRate_Respiration(
        filtered_heart_rate_model(n_states, rng),
        respiration_model(n_states, rng), rng)

    y_data = hmmds.applications.apnea.utilities.pattern_heart_rate_respiration_data(
        ['a'], common)
    # a list with a dict for each a-file

    model = develop.HMM(
        random_1d_prob(rng, 2),  # p_state_initial
        random_1d_prob(rng, 2),  # p_state_time_average
        random_conditional_prob(rng, (2, 2)),  # p_state2state
        y_model,
        rng)

    model.initialize_y_model(y_data)
    return model


@register
def C1(common, rng):
    """One state, no bundles, AR-4 for heart rate, single Gaussian for
    respiration

    """
    n_states = 1

    y_model = observation.FilteredHeartRate_Respiration(
        filtered_heart_rate_model(n_states, rng),
        respiration_model(n_states, rng), rng)

    name = 'c01'
    y_data = [
        hmmds.applications.apnea.utilities.heart_rate_respiration_data(
            name, common)
    ]

    model = develop.HMM(
        numpy.ones((1,), numpy.float64),  # p_state_initial
        numpy.ones((1,), numpy.float64),  # p_state_time_average
        numpy.ones((1, 1), numpy.float64),  # p_state2state
        y_model,
        rng)

    model.initialize_y_model(y_data)
    return model


@register
def Low(common, rng):
    n_states = 10

    bundle2state = {
        0: numpy.arange(6, dtype=numpy.int32),
        1: numpy.arange(6, 10, dtype=numpy.int32)
    }

    y_model = filtered_heart_rate_respiration_bundle_model(
        n_states, bundle2state, rng)

    # Without several c-names for initialization, training fails and
    # reports that the data is not plausible
    y_data = [
        hmmds.applications.apnea.utilities.heart_rate_respiration_bundle_data(
            name, common) for name in 'c01 c02 c03 c04 c05 c06'.split()
    ]

    model = develop.HMM(random_1d_prob(rng, n_states),
                        random_1d_prob(rng, n_states),
                        hmm.simple.Prob(P_SS_LowMedium).normalize(), y_model,
                        rng)
    assert model.p_state_initial.min() > 0
    model.initialize_y_model(y_data)
    assert model.p_state_initial.min() > 0
    return model


@register
def Medium(common, rng):
    return Low(common, rng)


@register
def High(common, rng):
    n_states = 14

    bundle2state = {
        0: numpy.arange(7, dtype=numpy.int32),
        1: numpy.arange(7, 14, dtype=numpy.int32)
    }

    y_model = filtered_heart_rate_respiration_bundle_model(
        n_states, bundle2state, rng)

    # Without several a-names for initialization, training fails and
    # reports that the data is not plausible
    y_data = [
        hmmds.applications.apnea.utilities.heart_rate_respiration_bundle_data(
            name, common) for name in 'a01 a02 a03 a04 a05 a06'.split()
    ]

    model = develop.HMM(random_1d_prob(rng, n_states),
                        random_1d_prob(rng, n_states),
                        hmm.simple.Prob(P_SS_High).normalize(), y_model, rng)
    model.initialize_y_model(y_data)
    return model


def main(argv=None):
    """
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser("Create and write/pickle an initial model")
    parser.add_argument('--root',
                        type=str,
                        default='../../../',
                        help='Path to level above hmmds')
    parser.add_argument('key', type=str, help='One of A2 C1 High Medium Low')
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    rng = numpy.random.default_rng()
    common = hmmds.applications.apnea.utilities.Common(args.root)

    # Run the function specified by args.key
    model = MODELS[args.key](common, rng)
    assert model.p_state_initial.min() > 0

    with open(args.write_path, 'wb') as _file:
        pickle.dump(model, _file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
